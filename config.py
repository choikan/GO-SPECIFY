# config.py
# 12



import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Keys
DATA_GO_KR_KEY = os.getenv("DATA_GO_KR_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Public API base
BASE_URL = os.getenv("DATA_GO_KR_BASE_URL", "https://apis.data.go.kr/1051000/recruitment").strip()

# Debug
DEBUG = os.getenv("DEBUG", "0").strip() in ("1", "true", "True", "YES", "yes")

# AI Mock
USE_AI_MOCK = os.getenv("USE_AI_MOCK", "0").strip() in ("1", "true", "True", "YES", "yes")
if not GOOGLE_API_KEY:
    USE_AI_MOCK = True
# Cache dir
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# caches
JOBS_CACHE_PATH = CACHE_DIR / "jobs_cache.json"

# ✅ favorites persistence
FAVORITES_PATH = CACHE_DIR / "favorites.json"

# ✅ saved roadmaps persistence
ROADMAPS_PATH = CACHE_DIR / "roadmaps.json"
