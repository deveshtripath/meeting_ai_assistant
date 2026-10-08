# AI Meeting Assistant

Turns a recorded English meeting (any audio or video file) into a transcript, minutes, key decisions
and action items.

```
audio/video ─▶ [1] faster-whisper ─▶ raw transcript
            ─▶ [2] LLM #1 (finds mis-heard technical terms) ─▶ refined transcript
            ─▶ [3] LLM #2 (documenter) ─▶ summary · minutes · decisions · action items (.md + .json)
```

## Quick start

1. Install Python 3.10+ and (recommended) ffmpeg.
2. `pip install -r requirements.txt`
3. Get a free key at https://console.groq.com
4. `streamlit run app.py`
5. On the main page, paste the key (or put `REFINER_API_KEY=...` in `.env`, see `.env.example`),
   then press **▶ Run on the sample**. The sample recording (`test_audio/test_meeting.mp3`) is on the page:
   play it, **download** it, or run it in one click. "What a correct result looks like" is right under it.

## What makes it fast

| Change | Effect |
|---|---|
| Refiner returns a short `{wrong → right}` list instead of re-typing the transcript; corrections are applied locally | LLM output drops from thousands of tokens to a few dozen; names/numbers/negations can't be rewritten (also checked in code) |
| Refiner and documenter chunks run in parallel (`LLM_WORKERS`) | Long meetings no longer scale linearly |
| Whisper: greedy decoding + batched inference + VAD, model preloaded at app start | Several times faster than beam-5 sequential decoding |
| Transcript cached by file hash | Re-running the same file skips transcription |
| Missing key / bad file detected before transcription | No wasted minutes |
| `gpt-oss` runs with low reasoning effort | Faster minutes generation |

For even more speed set `WHISPER_MODEL=base.en`; for more accuracy `medium.en`.

## What makes it work on more files

- Any audio/video container: decoded with PyAV, and automatically with the ffmpeg command line if PyAV fails.
- Quiet recordings are normalised; silent/too-short/no-audio-track files give a clear message.
- Automatic retry with back-off on API rate limits (429) and 5xx; clear messages for bad keys.
- Very long meetings (> `DOC_SINGLE_PASS_WORDS` words) are documented in parallel parts and merged.
- If the term-check step fails, the run continues with the raw transcript and shows a warning.
- Non-English: set `WHISPER_MODEL=small` and `WHISPER_LANGUAGE=auto`.

## Tests (no API key or audio models needed)
```bash
python -m unittest tests/test_pipeline.py -v
```

## Files
| File | Purpose |
|------|---------|
| `app.py` | One-page Streamlit UI |
| `pipeline.py` | Validation, audio loading, STT, refiner, documenter, orchestration |
| `prompts.py` / `schemas.py` / `config.py` | Prompts, Pydantic schema + Markdown renderer, settings |
| `test_audio/` | Sample meeting + expected results |

See `TECHNICAL_DESCRIPTION.md` for model roles and data flow.

## Deploy (GitHub -> Streamlit Community Cloud, free)

1. Create a GitHub repo and push **only the code** (the included `.gitignore` excludes `.venv`, models, `.env`):
   ```bash
   cd meeting_assistant
   git init && git add . && git commit -m "Meeting Assistant"
   git branch -M main
   git remote add origin https://github.com/<you>/meeting-assistant.git
   git push -u origin main
   ```
   `git count-objects -vH` should show a few MB. If it is huge, run `git rm -r --cached .venv` and commit again.
2. Go to https://share.streamlit.io -> **Create app** -> pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings -> Secrets**: paste the contents of `.streamlit/secrets.toml.example` with your real Groq key.
   (`packages.txt` installs ffmpeg automatically; `requirements.txt` installs the Python packages.)
4. Deploy. The Whisper model downloads once on first start (not stored in GitHub).