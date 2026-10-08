"""End-to-end pipeline: speech-to-text -> LLM #1 (refine) -> LLM #2 (document).

Every stage output is produced by a model at run time; nothing is hardcoded.

Speed design
  * STT: greedy decoding + batched inference + VAD; audio decoded once (PyAV, ffmpeg fallback);
    transcripts cached by file hash, so re-running the same file is instant.
  * Refiner: the LLM returns a short list of {wrong -> right} corrections (a few dozen tokens)
    instead of re-typing the whole transcript; chunks run in parallel; corrections are applied locally.
  * Documenter: one call for normal meetings, parallel map-reduce for very long ones.
  * Fails fast: API key / file problems are reported BEFORE the slow transcription starts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from pydantic import ValidationError

import config
import prompts
from schemas import MeetingRecord, MinuteSection, to_markdown


class PipelineError(Exception):
    """Raised with a user-friendly message when a stage cannot complete."""


@dataclass
class PipelineResult:
    raw_transcript: str = ""
    refined_transcript: str = ""
    corrections: list = field(default_factory=list)  # [{"wrong":..., "right":..., "count":...}]
    record: Optional[MeetingRecord] = None
    markdown: str = ""
    json_text: str = ""
    models: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)  # seconds per stage
    warnings: list = field(default_factory=list)
    cached_transcript: bool = False


StatusCb = Callable[[str, str], None]  # (stage, message)
ProgressCb = Callable[[float], None]  # 0..1 during transcription


def _noop(*_a, **_k) -> None:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- validation
def validate_audio_file(path: str, original_name: str | None = None) -> None:
    name = original_name or path
    ext = os.path.splitext(name)[1].lower()
    if ext and ext not in config.SUPPORTED_EXTENSIONS:
        raise PipelineError(
            f"Unsupported file type '{ext}'. Please upload an audio or video recording "
            f"({', '.join(config.SUPPORTED_EXTENSIONS)})."
        )
    if not os.path.exists(path):
        raise PipelineError("The uploaded file could not be found.")
    size = os.path.getsize(path)
    if size == 0:
        raise PipelineError("The uploaded file is empty (0 bytes).")
    if size > config.MAX_FILE_MB * 1024 * 1024:
        raise PipelineError(f"The file is larger than the {config.MAX_FILE_MB} MB limit.")


# --------------------------------------------------------------------------- audio loading
SAMPLE_RATE = 16000


def _ffmpeg_decode(path: str):
    """Fallback decoder: the ffmpeg CLI -> 16 kHz mono float32 numpy array."""
    import numpy as np

    exe = shutil.which("ffmpeg")
    if not exe:
        raise PipelineError(
            "The audio could not be decoded and ffmpeg is not installed as a fallback. "
            "Install ffmpeg (https://ffmpeg.org) or upload a .wav/.mp3 file."
        )
    cmd = [exe, "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise PipelineError("Decoding the file took too long.") from e
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "ignore").strip().splitlines()[-1:] or ["unknown error"]
        raise PipelineError(
            "The file could not be read as audio/video, or it has no audio track. "
            f"(ffmpeg: {err[0][:200]})"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(path: str):
    """Decode ANY audio/video file once into 16 kHz mono float32; light peak-normalisation for quiet recordings."""
    import numpy as np

    audio = None
    try:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
    except Exception:  # PyAV missing/incompatible/odd container -> ffmpeg CLI
        audio = None
    if audio is None or len(audio) == 0:
        audio = _ffmpeg_decode(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < SAMPLE_RATE // 2:
        raise PipelineError("The recording is shorter than half a second - there is nothing to transcribe.")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-3:
        raise PipelineError("The recording is silent (no audible speech).")
    if peak < 0.5:  # quiet recording: boost (capped) so Whisper hears it
        audio = audio * min(0.9 / peak, 8.0)
    return audio


# --------------------------------------------------------------------------- stage 1: STT
_whisper = None
_batched = None
_whisper_lock = threading.Lock()


def _resolve_device() -> Tuple[str, str]:
    device = config.WHISPER_DEVICE
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    ctype = config.WHISPER_COMPUTE_TYPE
    if ctype == "auto":
        ctype = "float16" if device == "cuda" else "int8"
    return device, ctype


def _get_whisper():
    global _whisper, _batched
    with _whisper_lock:
        if _whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise PipelineError("faster-whisper is not installed. Run: pip install -r requirements.txt") from e
            device, ctype = _resolve_device()
            _whisper = WhisperModel(
                config.WHISPER_MODEL, device=device, compute_type=ctype, cpu_threads=config.CPU_THREADS
            )
            try:  # faster-whisper >= 1.1: several speech segments decoded in parallel
                from faster_whisper import BatchedInferencePipeline

                _batched = BatchedInferencePipeline(model=_whisper)
            except ImportError:
                _batched = None
    return _whisper, _batched


def warmup() -> None:
    """Load the Whisper model in the background so the first run does not pay the loading cost."""
    try:
        _get_whisper()
    except Exception:
        pass


def _file_key(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    h.update(f"|{config.WHISPER_MODEL}|{config.WHISPER_LANGUAGE}|{config.WHISPER_BEAM_SIZE}".encode())
    return h.hexdigest()[:40]


def _cache_path(key: str) -> str:
    return os.path.join(config.CACHE_DIR, f"{key}.txt")


def _decode_segments(model, audio, *, batched: bool, vad: bool, duration: float, on_progress: ProgressCb) -> str:
    language = None if config.WHISPER_LANGUAGE == "auto" else config.WHISPER_LANGUAGE
    kwargs = dict(
        language=language,
        beam_size=max(1, config.WHISPER_BEAM_SIZE),
        initial_prompt=config.WHISPER_PROMPT or None,
        vad_filter=vad,
    )
    if vad:
        kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)
    if batched:
        kwargs["batch_size"] = config.WHISPER_BATCH_SIZE
    else:
        kwargs["condition_on_previous_text"] = False  # avoids repetition loops on long files
    segments, _info = model.transcribe(audio, **kwargs)
    parts = []
    for seg in segments:
        t = seg.text.strip()
        if t:
            parts.append(t)
        if duration > 0:
            on_progress(min(1.0, float(seg.end) / duration))
    return " ".join(parts).strip()


def transcribe(path: str, on_progress: ProgressCb = _noop) -> Tuple[str, bool]:
    """Return (transcript, served_from_cache)."""
    key = _file_key(path)
    cp = _cache_path(key)
    try:
        with open(cp, encoding="utf-8") as f:
            cached = f.read().strip()
        if cached:
            on_progress(1.0)
            return cached, True
    except OSError:
        pass

    audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    model, batched = _get_whisper()
    try:
        text = _decode_segments(
            batched or model, audio, batched=batched is not None, vad=True, duration=duration, on_progress=on_progress
        )
        if not text:  # VAD can drop speech buried in music/noise; retry once without it
            text = _decode_segments(model, audio, batched=False, vad=False, duration=duration, on_progress=on_progress)
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(
            "Transcription failed. The file may be corrupted or not a real recording. "
            f"(Details: {type(e).__name__}: {e})"
        ) from e
    if not text:
        raise PipelineError("No speech was detected in the recording (it may be silent or too noisy).")
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass
    return text, False


# --------------------------------------------------------------------------- LLM helper
def _client(base_url: str, api_key: str):
    local = "localhost" in base_url or "127.0.0.1" in base_url
    if not api_key and not local:
        raise PipelineError(
            "No API key set. Paste a free Groq key (console.groq.com) in the box on the main page, "
            "or put it in the .env file as REFINER_API_KEY."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise PipelineError("openai package is not installed. Run: pip install -r requirements.txt") from e
    return OpenAI(
        base_url=base_url, api_key=api_key or "not-needed", timeout=config.LLM_TIMEOUT_S, max_retries=config.LLM_RETRIES
    )


def _status_code(e: Exception) -> Optional[int]:
    return getattr(e, "status_code", None)


def _friendly(model: str, e: Exception) -> str:
    code = _status_code(e)
    if code in (401, 403):
        return f"The API key was rejected by the language model service ({model}). Please check the key."
    if code == 429:
        return f"The language model service is rate-limiting requests ({model}). Wait a minute and try again."
    if code == 404:
        return f"Model '{model}' was not found on the language model service. Check REFINER_MODEL / DOC_MODEL."
    return f"Language model call failed ({model}): {e}"


def _chat(client, model: str, messages: list, json_mode: bool = False) -> str:
    base = dict(model=model, messages=messages, temperature=0)
    full = dict(base)
    if json_mode:
        full["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model.lower() and config.DOC_REASONING_EFFORT:
        full["extra_body"] = {"reasoning_effort": config.DOC_REASONING_EFFORT}  # shorter hidden reasoning = faster
    attempts = [full]
    if "extra_body" in full:
        attempts.append({k: v for k, v in full.items() if k != "extra_body"})
    if len(full) > len(base):
        attempts.append(base)  # provider supports neither JSON mode nor reasoning_effort
    last: Optional[Exception] = None
    for kwargs in attempts:
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except Exception as e:
            last = e
            if _status_code(e) not in (400, 422):  # only unsupported-parameter errors are worth a retry here
                break
    raise PipelineError(_friendly(model, last)) from last


# --------------------------------------------------------------------------- text helpers
def _chunk_text(text: str, max_words: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start : end + 1])


def _run_parallel(fn, items: list, on_done: Callable[[int, int], None] = _noop) -> list:
    """Run fn over items with a small thread pool; results keep input order; callbacks run in the CALLING thread."""
    if len(items) <= 1:
        out = []
        for i, it in enumerate(items, 1):
            out.append(fn(it))
            on_done(i, len(items))
        return out
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max(1, min(config.LLM_WORKERS, len(items)))) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for n, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()
            on_done(n, len(items))
    return results


# --------------------------------------------------------------------------- stage 2: refine
_NEG = re.compile(
    r"\b(?:not|no|never|cannot|without|\w+n't|\w+n’t)\b", re.IGNORECASE
)


def _valid_correction(wrong: object, right: object) -> bool:
    if not isinstance(wrong, str) or not isinstance(right, str):
        return False
    wrong, right = wrong.strip(), right.strip()
    if not wrong or not right or wrong == right:
        return False
    if len(wrong.split()) > 6 or len(right.split()) > 6:
        return False
    if sorted(re.findall(r"\d+", wrong)) != sorted(re.findall(r"\d+", right)):
        return False  # never change numbers / dates / amounts
    if sorted(m.lower() for m in _NEG.findall(wrong)) != sorted(m.lower() for m in _NEG.findall(right)):
        return False  # never add/remove negations
    return True


def _pattern_for(wrong: str) -> re.Pattern:
    body = r"\s+".join(re.escape(w) for w in wrong.split())
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)


def apply_corrections(text: str, corrections: List[dict]) -> Tuple[str, List[dict]]:
    """Apply validated corrections locally (longest phrase first). Returns (new_text, applied_with_counts)."""
    seen, ordered = set(), []
    for c in corrections:
        if not isinstance(c, dict) or not _valid_correction(c.get("wrong"), c.get("right")):
            continue
        key = c["wrong"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append((c["wrong"].strip(), c["right"].strip()))
    ordered.sort(key=lambda p: -len(p[0]))
    applied = []
    for wrong, right in ordered:
        text, n = _pattern_for(wrong).subn(lambda _m, r=right: r, text)
        if n:
            applied.append({"wrong": wrong, "right": right, "count": n})
    return text, applied


def _find_corrections(client, chunk: str) -> List[dict]:
    reply = _chat(
        client,
        config.REFINER_MODEL,
        [
            {"role": "system", "content": prompts.REFINER_SYSTEM},
            {"role": "user", "content": prompts.REFINER_USER.format(chunk=chunk)},
        ],
        json_mode=True,
    )
    try:
        data = _extract_json(reply)
    except ValueError:
        return []  # unusable reply for this chunk -> no corrections (original text is kept)
    items = data.get("corrections") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []


def refine_transcript(raw: str, on_status: StatusCb = _noop, api_key: str | None = None) -> Tuple[str, List[dict]]:
    """Return (refined_text, applied_corrections). Never makes the pipeline fail: on LLM trouble, raw text is kept."""
    client = _client(config.REFINER_BASE_URL, api_key or config.REFINER_API_KEY)
    chunks = _chunk_text(raw, config.REFINE_CHUNK_WORDS)

    def work(ch: str):
        try:
            return _find_corrections(client, ch), None
        except PipelineError as e:
            return [], e

    results = _run_parallel(
        work, chunks, lambda d, n: on_status("refine", f"Checking technical terms ({d}/{n})…")
    )
    errors = [e for _c, e in results if e is not None]
    if errors and len(errors) == len(results):
        raise errors[0]
    all_corrections = [c for cs, _e in results for c in cs]
    return apply_corrections(raw, all_corrections)


# --------------------------------------------------------------------------- stage 3: document
def _document_once(client, transcript: str, on_status: StatusCb = _noop, part_note: str = "") -> MeetingRecord:
    messages = [
        {"role": "system", "content": prompts.DOCUMENTER_SYSTEM},
        {"role": "user", "content": prompts.DOCUMENTER_USER.format(transcript=transcript, part_note=part_note)},
    ]
    reply = _chat(client, config.DOC_MODEL, messages, json_mode=True)
    for attempt in range(2):  # one initial parse + one repair retry
        try:
            return MeetingRecord.model_validate(_extract_json(reply))
        except (ValueError, ValidationError) as err:
            if attempt == 1:
                raise PipelineError(
                    "The documentation model returned malformed output twice. Please try again."
                ) from err
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": prompts.JSON_REPAIR_USER.format(error=str(err)[:500])},
            ]
            reply = _chat(client, config.DOC_MODEL, messages, json_mode=True)
    raise PipelineError("Unexpected failure while generating the meeting record.")  # pragma: no cover


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _merge_records(client, records: List[MeetingRecord]) -> MeetingRecord:
    minutes: dict = {}
    for r in records:
        for sec in r.minutes:
            minutes.setdefault(_norm(sec.topic), MinuteSection(topic=sec.topic, points=[])).points.extend(sec.points)
    decisions, d_seen = [], set()
    for r in records:
        for d in r.decisions:
            if _norm(d.decision) not in d_seen:
                d_seen.add(_norm(d.decision))
                decisions.append(d)
    actions, a_seen = [], set()
    for r in records:
        for a in r.action_items:
            if _norm(a.task) not in a_seen:
                a_seen.add(_norm(a.task))
                actions.append(a)
    joined = "\n".join(f"{i}. {r.summary}" for i, r in enumerate(records, 1))
    try:
        summary = _chat(
            client,
            config.DOC_MODEL,
            [
                {"role": "system", "content": prompts.SUMMARY_MERGE_SYSTEM},
                {"role": "user", "content": prompts.SUMMARY_MERGE_USER.format(summaries=joined)},
            ],
        ) or " ".join(r.summary for r in records)
    except PipelineError:
        summary = " ".join(r.summary for r in records)
    return MeetingRecord(summary=summary, minutes=list(minutes.values()), decisions=decisions, action_items=actions)


def generate_record(refined: str, on_status: StatusCb = _noop, api_key: str | None = None) -> MeetingRecord:
    client = _client(config.DOC_BASE_URL, api_key or config.DOC_API_KEY)
    if len(refined.split()) <= config.DOC_SINGLE_PASS_WORDS:
        return _document_once(client, refined, on_status)
    parts = _chunk_text(refined, config.DOC_CHUNK_WORDS)
    total = len(parts)
    notes = [
        f"\n(This is part {i} of {total} of a longer meeting; document only what is in this part.)"
        for i in range(1, total + 1)
    ]
    records = _run_parallel(
        lambda ip: _document_once(client, ip[0], on_status, ip[1]),
        list(zip(parts, notes)),
        lambda d, n: on_status("document", f"Writing minutes ({d}/{n} parts)…"),
    )
    on_status("document", "Merging parts…")
    return _merge_records(client, records)


# --------------------------------------------------------------------------- orchestration
def run_pipeline(
    path: str,
    original_name: str | None = None,
    on_status: StatusCb = _noop,
    api_key: str | None = None,
    on_progress: ProgressCb = _noop,
) -> PipelineResult:
    res = PipelineResult(
        models={
            "speech_to_text": f"faster-whisper / {config.WHISPER_MODEL}",
            "llm_1_refiner": config.REFINER_MODEL,
            "llm_2_documenter": config.DOC_MODEL,
        }
    )
    on_status("validate", "Checking the file and API key…")
    validate_audio_file(path, original_name)
    # Fail fast: don't spend minutes transcribing only to discover a missing key.
    _client(config.REFINER_BASE_URL, api_key or config.REFINER_API_KEY)
    _client(config.DOC_BASE_URL, api_key or config.DOC_API_KEY)

    t0 = time.perf_counter()
    on_status("transcribe", "Transcribing…")
    res.raw_transcript, res.cached_transcript = transcribe(path, on_progress)
    res.timings["transcribe"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    on_status("refine", "Checking technical terms…")
    try:
        res.refined_transcript, res.corrections = refine_transcript(res.raw_transcript, on_status, api_key)
    except PipelineError as e:  # refinement is an enhancement; don't lose the whole run over it
        res.refined_transcript, res.corrections = res.raw_transcript, []
        res.warnings.append(f"Term refinement skipped: {e}")
    res.timings["refine"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    on_status("document", "Writing minutes, decisions and action items…")
    res.record = generate_record(res.refined_transcript, on_status, api_key)
    res.timings["document"] = time.perf_counter() - t0

    res.markdown = to_markdown(res.record)
    res.json_text = json.dumps({"models": res.models, **res.record.model_dump()}, indent=2, ensure_ascii=False)
    on_status("done", "Done.")
    return res