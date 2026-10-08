"""Streamlit UI for the AI Meeting Assistant.  Run:  streamlit run app.py"""
import io
import json
import os
import tempfile
import threading
import time
import zipfile

import streamlit as st

import config
import pipeline
from pipeline import PipelineError, PipelineResult, run_pipeline
from schemas import to_markdown

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "test_audio", "test_meeting.mp3")
EXPECTED_PATH = os.path.join(HERE, "test_audio", "expected_results.md")

st.set_page_config(page_title="AI Meeting Assistant", page_icon="📝", layout="centered")


@st.cache_resource
def _warmup():
    """Load the speech model in the background once per server, so the first run starts fast."""
    t = threading.Thread(target=pipeline.warmup, daemon=True)
    t.start()
    return t


_warmup()

st.title("📝 AI Meeting Assistant")
st.write("Upload a meeting recording and get the transcript, minutes, decisions and action items.")

# ------------------------------------------------------------------ API key (only asked when not configured)
KEY_NAMES = ("REFINER_API_KEY", "GROQ_API_KEY", "DOC_API_KEY", "API_KEY")


def _clean(val) -> str:
    return str(val or "").strip().strip("'\"").strip()


def _flatten(d, prefix=""):
    """Yield (name, value) for every secret, including ones nested under a [section]."""
    for k, v in d.items():
        if hasattr(v, "items"):
            yield from _flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def _configured_key() -> tuple[str, str, list[str]]:
    """Return (key, where_found, secret_names_seen). Env/.env first, then Streamlit secrets."""
    for name in KEY_NAMES:
        val = _clean(os.getenv(name))
        if val:
            return val, f"environment variable {name}", []

    names, error = [], ""
    try:
        items = list(_flatten(st.secrets))
    except Exception as e:  # no secrets at all, or invalid TOML in the Secrets box
        items, error = [], type(e).__name__  # name only: the message can echo the secret text
    names = [n for n, _ in items]

    wanted = {n.lower() for n in KEY_NAMES}
    # exact name (top level or inside a section), case-insensitive
    for full, val in items:
        if full.split(".")[-1].lower() in wanted and _clean(val):
            return _clean(val), f"secret {full}", names
    # any other secret that looks like a Groq / API key
    for full, val in items:
        v, low = _clean(val), full.lower()
        if v and (v.startswith("gsk_") or "groq" in low or ("api" in low and "key" in low)):
            return v, f"secret {full}", names
    return "", error, names


api_key, key_source, secret_names = _configured_key()
if api_key:
    st.caption("🔑 Using the API key configured on the server.")
else:
    st.warning(
        "No API key is configured on the server. Paste a free Groq key below "
        "(site owner: add REFINER_API_KEY in Streamlit → Settings → Secrets, then reboot the app)."
    )
    st.caption(
        f"Debug: secrets visible to the app: {', '.join(secret_names) or 'none'}"
        + (
            f" · secrets could not be read ({key_source}). Check the Secrets box is valid TOML, "
            'with quotes around the key: REFINER_API_KEY = "gsk_..."'
            if key_source else ""
        )
    )
    api_key = st.text_input(
        "Free Groq API key",
        type="password",
        placeholder="Paste your key here",
        help="Get one in a minute at console.groq.com. It is only used for this session.",
    ).strip()

# ------------------------------------------------------------------ input: sample or own file
job = None  # (path, display_name, is_temp)

with st.container(border=True):
    st.subheader("1 · Try the sample")
    if os.path.exists(SAMPLE_PATH):
        st.audio(SAMPLE_PATH)
        c1, c2 = st.columns(2)
        with open(SAMPLE_PATH, "rb") as f:
            c1.download_button("⬇ Download sample audio", f.read(), "test_meeting.mp3", "audio/mpeg", use_container_width=True)
        if c2.button("▶ Run on the sample", type="primary", use_container_width=True):
            job = (SAMPLE_PATH, "test_meeting.mp3", False)
        if os.path.exists(EXPECTED_PATH):
            with st.expander("What a correct result looks like"):
                with open(EXPECTED_PATH, encoding="utf-8") as f:
                    st.markdown(f.read())
    else:
        st.info("Sample audio not found in test_audio/.")

with st.container(border=True):
    st.subheader("2 · Or use your own recording")
    uploaded = st.file_uploader(
        "Audio or video file (mp3, wav, m4a, mp4, …)", label_visibility="collapsed", type=None
    )
    if st.button("▶ Process my file", type="primary", disabled=uploaded is None):
        suffix = os.path.splitext(uploaded.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
        job = (tmp.name, uploaded.name, True)

# ------------------------------------------------------------------ run
if job and not api_key:
    st.error("Please enter a Groq API key first (or configure REFINER_API_KEY in the app's Secrets).")
    if job[2]:
        try:
            os.unlink(job[0])
        except OSError:
            pass
    job = None

if job:
    path, name, is_temp = job
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    bar = st.progress(0.0, text="Starting…")
    marks = {"validate": 0.02, "transcribe": 0.05, "refine": 0.65, "document": 0.8, "done": 1.0}
    started = time.perf_counter()

    def on_status(stage: str, msg: str):
        bar.progress(marks.get(stage, 0.0), text=msg)

    def on_progress(frac: float):
        bar.progress(0.05 + 0.6 * min(1.0, frac), text=f"Transcribing… {int(frac * 100)}%")

    try:
        result = run_pipeline(path, original_name=name, on_status=on_status, api_key=api_key, on_progress=on_progress)
        st.session_state["result"] = result
        st.session_state["source_name"] = name
        st.session_state["elapsed"] = time.perf_counter() - started
    except PipelineError as e:
        st.session_state["error"] = str(e)
    except Exception as e:  # last-resort guard so the UI never shows a raw traceback
        st.session_state["error"] = f"Unexpected error: {type(e).__name__}: {e}"
    finally:
        bar.empty()
        if is_temp:
            try:
                os.unlink(path)
            except OSError:
                pass

if "error" in st.session_state:
    st.error(st.session_state["error"])

# ------------------------------------------------------------------ results
result: PipelineResult | None = st.session_state.get("result")
if result and result.record:
    rec = result.record
    stem = os.path.splitext(st.session_state.get("source_name", "meeting"))[0]
    t = result.timings
    note = " (transcript from cache)" if result.cached_transcript else ""
    st.success(
        f"Done in {st.session_state.get('elapsed', sum(t.values())):.0f}s — transcribe {t.get('transcribe', 0):.0f}s · "
        f"terms {t.get('refine', 0):.0f}s · minutes {t.get('document', 0):.0f}s{note}"
    )
    for w in result.warnings:
        st.warning(w)

    tab_rec, tab_tr, tab_dl = st.tabs(["Meeting record", "Transcript", "Download"])

    with tab_rec:
        st.subheader("Summary")
        st.write(rec.summary)

        st.subheader("Key decisions")
        if rec.decisions:
            for i, d in enumerate(rec.decisions, 1):
                st.markdown(f"**{i}. {d.decision}**" + (f"  \n{d.context}" if d.context else ""))
        else:
            st.info("No decisions were made in this meeting.")

        st.subheader("Action items")
        if rec.action_items:
            st.table(
                [
                    {"#": i, "Task": a.task, "Owner": a.owner, "Deadline": a.deadline}
                    for i, a in enumerate(rec.action_items, 1)
                ]
            )
        else:
            st.info("No action items were assigned in this meeting.")

        with st.expander("Detailed minutes"):
            for sec in rec.minutes:
                st.markdown(f"**{sec.topic}**")
                for p in sec.points:
                    st.markdown(f"- {p}")

    with tab_tr:
        if result.corrections:
            fixes = ", ".join(f"{c['wrong']} → **{c['right']}**" for c in result.corrections)
            st.caption(f"Technical terms fixed: {fixes}")
        else:
            st.caption("No technical-term corrections were needed.")
        st.text_area("Refined transcript", result.refined_transcript, height=350)
        with st.expander("Raw speech-to-text output"):
            st.text_area("Raw transcript", result.raw_transcript, height=250, label_visibility="collapsed")

    with tab_dl:
        decisions_md = (
            "\n".join(f"{i}. {d.decision} — {d.context}" for i, d in enumerate(rec.decisions, 1)) or "None."
        )
        files = [
            ("Raw transcript", result.raw_transcript, f"{stem}_raw_transcript.txt", "text/plain"),
            ("Refined transcript", result.refined_transcript, f"{stem}_refined_transcript.txt", "text/plain"),
            ("Meeting record (.md)", to_markdown(rec, title="Meeting Minutes"), f"{stem}_meeting_record.md", "text/markdown"),
            ("Meeting record (.json)", result.json_text, f"{stem}_meeting_record.json", "application/json"),
            ("Key decisions", decisions_md, f"{stem}_decisions.txt", "text/plain"),
            (
                "Action items (.json)",
                json.dumps([a.model_dump() for a in rec.action_items], indent=2),
                f"{stem}_action_items.json",
                "application/json",
            ),
        ]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for _label, data, fname, _mime in files:
                z.writestr(fname, data)
        st.download_button(
            "⬇ Download everything (.zip)", buf.getvalue(), f"{stem}_meeting_package.zip", "application/zip",
            type="primary", key="zip_all",
        )
        st.caption("Or pick a single file:")
        cols = st.columns(2)
        for i, (label, data, fname, mime) in enumerate(files):
            cols[i % 2].download_button(f"⬇ {label}", data, fname, mime, key=fname, use_container_width=True)

st.divider()
st.caption(
    f"Models: {config.WHISPER_MODEL} (speech) · {config.REFINER_MODEL} (term check) · {config.DOC_MODEL} (minutes)"
)