"""Unit tests with a mocked LLM client (orchestration, corrections, repair, map-reduce, decoding)."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import pipeline  # noqa: E402
from schemas import ActionItem, MeetingRecord, to_markdown  # noqa: E402


def fake_client(replies):
    """Replies are consumed in order (str), or a callable(messages)->str."""
    it = iter(replies) if not callable(replies) else None
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        content = replies(kwargs["messages"]) if it is None else next(it)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    c = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    c.calls = calls
    return c


GOOD = json.dumps(
    {
        "summary": "Team discussed the release.",
        "minutes": [{"topic": "Release", "points": ["Ship on Friday was proposed."]}],
        "decisions": [{"decision": "Use PostgreSQL", "context": "Agreed by all"}],
        "action_items": [
            {"task": "Write migration script", "owner": "Priya", "deadline": None},
            {"task": "Update docs", "owner": "N/A", "deadline": "next Monday"},
        ],
    }
)


class Tests(unittest.TestCase):
    def test_validate_rejects_bad_extension_and_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
        with self.assertRaises(pipeline.PipelineError):
            pipeline.validate_audio_file(f.name, "notes.txt")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
            pass
        with self.assertRaises(pipeline.PipelineError) as cm:
            pipeline.validate_audio_file(f2.name, "x.wav")
        self.assertIn("empty", str(cm.exception))

    def test_missing_owner_deadline_become_unspecified(self):
        a = ActionItem(task="t", owner=None, deadline="TBD")
        self.assertEqual((a.owner, a.deadline), ("unspecified", "unspecified"))

    def test_generate_record_with_json_repair(self):
        client = fake_client(["not json at all", GOOD])
        with mock.patch.object(pipeline, "_client", return_value=client):
            rec = pipeline.generate_record("some transcript")
        self.assertEqual(rec.action_items[0].deadline, "unspecified")
        self.assertEqual(rec.action_items[1].owner, "unspecified")

    def test_generate_record_fails_after_two_bad_replies(self):
        client = fake_client(["bad", "still bad"])
        with mock.patch.object(pipeline, "_client", return_value=client):
            with self.assertRaises(pipeline.PipelineError):
                pipeline.generate_record("x")

    # ---- refiner: corrections applied locally, with guards
    def test_corrections_applied_case_insensitive_word_boundary(self):
        text = "We use Cooper Netties and post gress. Postgresql stays. Costs 92 percent."
        out, applied = pipeline.apply_corrections(
            text,
            [{"wrong": "cooper netties", "right": "Kubernetes"}, {"wrong": "post gress", "right": "PostgreSQL"}],
        )
        self.assertEqual(out, "We use Kubernetes and PostgreSQL. Postgresql stays. Costs 92 percent.")
        self.assertEqual({a["right"] for a in applied}, {"Kubernetes", "PostgreSQL"})

    def test_corrections_guard_numbers_negations_and_junk(self):
        text = "We are not going to upgrade. Budget is 92 percent. Call Priya."
        bad = [
            {"wrong": "not going", "right": "going"},  # removes negation
            {"wrong": "92 percent", "right": "29 percent"},  # changes number
            {"wrong": "Priya", "right": None},  # junk
            {"wrong": "a b c d e f g", "right": "x"},  # too long
            "garbage",
        ]
        out, applied = pipeline.apply_corrections(text, bad)
        self.assertEqual(out, text)
        self.assertEqual(applied, [])

    def test_refine_uses_short_correction_list_and_keeps_text(self):
        raw = "We will move to post gress. Deploy on cooper netties. Meera shares pie torch results."
        reply = json.dumps(
            {
                "corrections": [
                    {"wrong": "post gress", "right": "PostgreSQL"},
                    {"wrong": "cooper netties", "right": "Kubernetes"},
                    {"wrong": "pie torch", "right": "PyTorch"},
                ]
            }
        )
        client = fake_client([reply])
        with mock.patch.object(pipeline, "_client", return_value=client):
            out, applied = pipeline.refine_transcript(raw)
        self.assertEqual(out, "We will move to PostgreSQL. Deploy on Kubernetes. Meera shares PyTorch results.")
        self.assertEqual(len(applied), 3)
        self.assertEqual(len(client.calls), 1)

    def test_refine_bad_reply_keeps_original(self):
        raw = "Nothing wrong here."
        with mock.patch.object(pipeline, "_client", return_value=fake_client(["I cannot do that"])):
            out, applied = pipeline.refine_transcript(raw)
        self.assertEqual((out, applied), (raw, []))

    def test_refine_parallel_many_chunks(self):
        raw = " ".join("Sentence number %d mentions cooper netties." % i for i in range(400))
        client = fake_client(lambda m: json.dumps({"corrections": [{"wrong": "cooper netties", "right": "Kubernetes"}]}))
        with mock.patch.object(pipeline, "_client", return_value=client), mock.patch.object(config, "REFINE_CHUNK_WORDS", 300):
            out, applied = pipeline.refine_transcript(raw)
        self.assertGreater(len(client.calls), 1)
        self.assertNotIn("cooper", out)
        self.assertEqual(applied[0]["count"], 400)

    def test_chunking(self):
        text = " ".join(["Sentence number %d is here." % i for i in range(300)])
        chunks = pipeline._chunk_text(text, 100)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), text)

    # ---- documenter
    def test_long_transcript_uses_map_reduce_and_dedupes(self):
        part = json.dumps(
            {
                "summary": "Part summary.",
                "minutes": [{"topic": "Release", "points": ["p"]}],
                "decisions": [{"decision": "Use PostgreSQL", "context": ""}],
                "action_items": [{"task": "Write migration script", "owner": "Priya", "deadline": "Friday"}],
            }
        )

        def reply(messages):
            return "Merged overview." if "partial summaries" in messages[0]["content"].lower() else part

        client = fake_client(reply)
        long_text = " ".join("Sentence %d goes here." % i for i in range(3000))
        with mock.patch.object(pipeline, "_client", return_value=client), mock.patch.object(config, "DOC_SINGLE_PASS_WORDS", 1000), mock.patch.object(config, "DOC_CHUNK_WORDS", 3000):
            rec = pipeline.generate_record(long_text)
        self.assertEqual(rec.summary, "Merged overview.")
        self.assertEqual(len(rec.decisions), 1)
        self.assertEqual(len(rec.action_items), 1)
        self.assertEqual(len(rec.minutes), 1)
        self.assertGreater(len(rec.minutes[0].points), 1)

    def test_chat_falls_back_when_provider_rejects_extra_params(self):
        class BadParam(Exception):
            status_code = 400

        seen = []

        def create(**kw):
            seen.append(sorted(kw))
            if "response_format" in kw or "extra_body" in kw:
                raise BadParam("unsupported")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(pipeline._chat(client, "openai/gpt-oss-120b", [], json_mode=True), "ok")
        self.assertEqual(len(seen), 3)

    def test_chat_auth_error_is_friendly_and_not_retried(self):
        class Auth(Exception):
            status_code = 401

        n = []

        def create(**kw):
            n.append(1)
            raise Auth("bad key")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaises(pipeline.PipelineError) as cm:
            pipeline._chat(client, "m", [], json_mode=True)
        self.assertIn("key", str(cm.exception))
        self.assertEqual(len(n), 1)

    def test_markdown_and_json_agree(self):
        rec = MeetingRecord.model_validate_json(GOOD)
        md = to_markdown(rec)
        self.assertIn("Use PostgreSQL", md)
        self.assertIn("Write migration script | Priya | unspecified", md)

    # ---- orchestration
    def test_missing_api_key_fails_before_transcription(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFFfake")
        with mock.patch.object(config, "REFINER_API_KEY", ""), mock.patch.object(config, "DOC_API_KEY", ""), mock.patch.object(
            pipeline, "transcribe"
        ) as tr:
            with self.assertRaises(pipeline.PipelineError) as cm:
                pipeline.run_pipeline(f.name, "m.wav")
        tr.assert_not_called()
        self.assertIn("API key", str(cm.exception))

    def test_full_pipeline_with_mocks(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFFfake")
        raw = "We will use post grass for the database. Priya will write the migration script."
        refine_reply = json.dumps({"corrections": [{"wrong": "post grass", "right": "PostgreSQL"}]})
        client = fake_client([refine_reply, GOOD])
        with mock.patch.object(pipeline, "transcribe", return_value=(raw, False)), mock.patch.object(
            pipeline, "_client", return_value=client
        ):
            res = pipeline.run_pipeline(f.name, "m.wav", api_key="k")
        self.assertEqual(res.raw_transcript, raw)
        self.assertIn("PostgreSQL", res.refined_transcript)
        self.assertEqual(json.loads(res.json_text)["decisions"][0]["decision"], "Use PostgreSQL")
        self.assertEqual(set(res.timings), {"transcribe", "refine", "document"})

    def test_refiner_outage_does_not_kill_run(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFFfake")
        calls = {"n": 0}

        def reply(messages):
            calls["n"] += 1
            if "proofreader" in messages[0]["content"]:
                raise_exc = pipeline.PipelineError("down")
                raise raise_exc
            return GOOD

        client = fake_client(reply)
        with mock.patch.object(pipeline, "transcribe", return_value=("hello there team.", False)), mock.patch.object(
            pipeline, "_client", return_value=client
        ):
            res = pipeline.run_pipeline(f.name, "m.wav", api_key="k")
        self.assertEqual(res.refined_transcript, "hello there team.")
        self.assertTrue(res.warnings)
        self.assertIsNotNone(res.record)


class DecodeTests(unittest.TestCase):
    SAMPLE = os.path.join(ROOT, "test_audio", "test_meeting.mp3")

    @unittest.skipUnless(shutil_which := __import__("shutil").which("ffmpeg"), "ffmpeg not installed")
    def test_ffmpeg_fallback_decodes_sample(self):
        audio = pipeline._ffmpeg_decode(self.SAMPLE)
        self.assertGreater(len(audio) / 16000, 100)  # ~143 s

    @unittest.skipUnless(__import__("shutil").which("ffmpeg"), "ffmpeg not installed")
    def test_load_audio_works_without_pyav_and_rejects_garbage(self):
        audio = pipeline.load_audio(self.SAMPLE)  # faster_whisper may be absent -> ffmpeg path
        self.assertEqual(audio.dtype.name, "float32")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"this is not audio at all" * 100)
        with self.assertRaises(pipeline.PipelineError):
            pipeline.load_audio(f.name)


if __name__ == "__main__":
    unittest.main()