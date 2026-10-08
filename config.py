"""Configuration. All values can be overridden through environment variables or a .env file."""
import os
import tempfile

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# ---- Speech-to-text (stage 1) ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")  # tiny.en / base.en / small.en / medium.en
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto / cpu / cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # auto -> int8 on CPU, float16 on GPU
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")  # "en", or "auto" with a multilingual model (e.g. "small")
WHISPER_BEAM_SIZE = _int("WHISPER_BEAM_SIZE", 1)  # 1 = greedy (much faster, ~same quality on clean speech)
WHISPER_BATCH_SIZE = _int("WHISPER_BATCH_SIZE", 8)
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "A business meeting. Technical terms, product names and acronyms are written correctly.",
)
CPU_THREADS = _int("CPU_THREADS", min(8, os.cpu_count() or 4))

# ---- LLM #1: transcript refiner (stage 2) ----
REFINER_BASE_URL = os.getenv("REFINER_BASE_URL", "https://api.groq.com/openai/v1")
REFINER_API_KEY = os.getenv("REFINER_API_KEY", "")
REFINER_MODEL = os.getenv("REFINER_MODEL", "llama-3.3-70b-versatile")

# ---- LLM #2: meeting documenter (stage 3) -- intentionally a DIFFERENT model ----
DOC_BASE_URL = os.getenv("DOC_BASE_URL", REFINER_BASE_URL)
DOC_API_KEY = os.getenv("DOC_API_KEY", REFINER_API_KEY)
DOC_MODEL = os.getenv("DOC_MODEL", "openai/gpt-oss-120b")
DOC_REASONING_EFFORT = os.getenv("DOC_REASONING_EFFORT", "low")  # only sent to gpt-oss models

# ---- Limits / speed ----
SUPPORTED_EXTENSIONS = (
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma", ".amr",
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp",
)
MAX_FILE_MB = _int("MAX_FILE_MB", 200)
REFINE_CHUNK_WORDS = _int("REFINE_CHUNK_WORDS", 1200)  # refiner only returns a short correction list
DOC_SINGLE_PASS_WORDS = _int("DOC_SINGLE_PASS_WORDS", 6000)  # longer transcripts use parallel map-reduce
DOC_CHUNK_WORDS = _int("DOC_CHUNK_WORDS", 3500)
LLM_WORKERS = _int("LLM_WORKERS", 3)  # parallel LLM calls
LLM_TIMEOUT_S = _int("LLM_TIMEOUT_S", 90)
LLM_RETRIES = _int("LLM_RETRIES", 4)  # automatic back-off on 429 / 5xx
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(tempfile.gettempdir(), "meeting_assistant_cache"))