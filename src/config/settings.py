"""Configuration settings for SentimentNusa."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Models directory - where Hugging Face models will be stored locally
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Set Hugging Face cache to use local models folder
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HF_HOME"] = str(MODELS_DIR)

# API Configuration
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")

# Apify Actor IDs for different platforms
# These are loaded from environment variables with defaults
APIFY_ACTORS = {
    "tiktok": os.getenv("APIFY_TIKTOK_ACTOR", ""),
    "instagram": os.getenv("APIFY_INSTAGRAM_ACTOR", ""),
}

# Scraping Configuration (default values)
DEFAULT_COMMENTS_PER_POST = 10
DEFAULT_MAX_REPLIES = 3
DEFAULT_RESULTS_PER_PAGE = 3

# Sentiment Analysis Configuration
SENTIMENT_MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"
MAX_TEXT_LENGTH = 512

# Visualization Configuration
DEFAULT_FIGURE_SIZE = (12, 6)
WORD_CLOUD_WIDTH = 800
WORD_CLOUD_HEIGHT = 400

# Indonesian stopwords for text processing
INDONESIAN_STOPWORDS = [
    "yg", "yang", "di", "dan", "itu", "ini", "ada", "ke", "dari",
    "adalah", "untuk", "ga", "gak", "nggak", "tidak", "bgt", "banget",
    "bgtt", "bgus", "bagus", "sama", "dengan", "oleh", "pada", "atau",
    "karena", "jika", "jadi", "akan", "bisa", "sudah", "saya", "aku",
    "kamu", "dia", "kita", "mereka", "nya", "lah", "sih", "kan", "dong"
]

# Text normalization dictionary
NORMALIZATION_DICT = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "bgus": "bagus",
    "bgs": "bagus",
    "tdk": "tidak",
    "tdak": "tidak",
    "dg": "dengan",
    "dgn": "dengan",
    "krn": "karena",
    "karna": "karena",
    "bs": "bisa",
    "bisa": "bisa",
    "jg": "juga",
    "jga": "juga",
    "sdh": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "sy": "saya",
    "syaa": "saya",
    "aq": "aku",
    "gwe": "aku",
    "gua": "aku",
    "gw": "aku",
    "km": "kamu",
    "kmu": "kamu",
    "loe": "kamu",
    "lo": "kamu",
    "lu": "kamu",
}
