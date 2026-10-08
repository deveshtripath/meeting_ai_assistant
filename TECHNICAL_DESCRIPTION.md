# Technical Description

## Models and roles
| Stage | Model (default) | Role |
|-------|-----------------|------|
| 1. Speech-to-text | `faster-whisper` `small.en` (CTranslate2 build of OpenAI Whisper), local, VAD-filtered | Converts uploaded audio into the **raw transcript** |
| 2. Language model #1 – *Refiner* | `llama-3.3-70b-versatile` | Fixes plausible ASR errors in technical terms, acronyms and domain vocabulary without altering names, numbers, negations or commitments. Produces the **refined transcript** |
| 3. Language model #2 – *Documenter* | `openai/gpt-oss-120b` | Reads only the refined transcript and emits a strict-JSON **meeting record**: summary, topic-wise minutes, decisions, action items |

Models are configurable in `.env`; the two LLM roles are separate functions, separate prompts and
use separate model settings (`REFINER_*` vs `DOC_*`).

## Data flow
1. **Validation** – extension whitelist, non-empty, size limit; API key checked up front.
2. **Transcription** – audio/video decoded once to 16 kHz mono (PyAV, ffmpeg CLI fallback), `faster-whisper` greedy + batched + VAD → `raw_transcript`. Result cached by file hash.
3. **Refinement** – the transcript is split into ~1200-word chunks; LLM #1 (temperature 0, JSON) returns only a list of `{wrong, right}` corrections for mis-heard technical terms. Chunks run in parallel. Corrections are validated in code (numbers and negations must be identical, max 6 words) and applied locally with word boundaries; the raw transcript is retained untouched.
4. **Documentation** – `refined_transcript` is sent to LLM #2 (temperature 0, JSON mode, low reasoning effort). Output is validated with Pydantic; on failure the model is shown the error and asked once to repair it. Transcripts over ~6000 words are split, documented in parallel and merged (de-duplicated decisions/actions, merged summary).
5. **Rendering** – the validated `MeetingRecord` is rendered to Markdown and dumped to JSON, so both are identical in content.

## Anti-hallucination design
- Prompt rules: use only transcript content; proposal ≠ decision; unassigned suggestion ≠ task; owner/deadline only if explicit, else `unspecified`.
- Schema validator normalises any "none / n/a / tbd / null / empty" to `unspecified`.
- Temperature 0 on both LLM stages.
- Refiner cannot rewrite text: it only proposes term corrections, and code rejects any that alter numbers or negations.

## Outputs
Raw transcript, refined transcript, minutes (with summary), key decisions, action items — displayed in tabs and downloadable individually, plus the combined `.md` and `.json` records.
