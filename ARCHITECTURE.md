# SentimentNusa — Architecture

## 1. Goals & Non-Goals

### Goals (what we WILL build)
- Scrape comments and posts from TikTok, Instagram, Facebook, and X (Twitter) using Apify actors
- Perform Indonesian-language sentiment analysis using a pre-trained RoBERTa model
- Provide an interactive web UI (Gradio) for non-technical users to run analyses
- Generate visualizations: sentiment distribution charts and word clouds
- Export results to Excel (.xlsx) and raw JSON for downstream use
- Persist raw extraction data locally for reproducibility and debugging

### Non-Goals (what we WON'T build)
- Real-time streaming or webhook-based ingestion (batch-only)
- User authentication / multi-tenancy (single-user desktop/Hugging Face Space app)
- Custom model training or fine-tuning (uses off-the-shelf HF model)
- Persistent database (file-based JSON + Excel output only)
- English or other language support as first-class (Indonesian optimized)

---

## 2. Core Principles

1. **Simplicity over scalability** — This is a research/analytics tool, not a production pipeline. File-based storage and in-memory processing are acceptable.
2. **Indonesian-first NLP** — Text preprocessing, slang normalization, and stopword removal are tuned for Indonesian social media text.
3. **Platform abstraction** — Each social platform has its own Apify actor configuration, but the scraper exposes a unified interface so adding a new platform only requires a new method + actor mapping.
4. **User-friendly packaging** — Distributed as a single `app.py` Gradio application runnable via `uv run app.py` or deployed to Hugging Face Spaces.
5. **No secrets in code** — API tokens are injected via environment variables or the Gradio UI; `.env` is gitignored.

---

## 3. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User (Browser)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Gradio Web Interface (app.py)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Sidebar      │  │ URL/Keyword  │  │ Analyze Button   │  │
│  │ - Token      │  │ Input        │  │ - Scrape         │  │
│  │ - Platform   │  │              │  │ - Preprocess     │  │
│  │ - Limits     │  │              │  │ - Visualize      │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              src/core/scraper.py                             │
│         SocialMediaScraper (Apify Client)                    │
│  ┌─────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────┐  │
│  │ TikTok  │ │ Instagram │ │ Facebook  │ │ X (Search)   │  │
│  │ Actor   │ │ Actor     │ │ Actor     │ │ Actor        │  │
│  └─────────┘ └───────────┘ └───────────┘ └──────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ Raw JSON items (List[Dict])
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           src/core/preprocessor.py                           │
│              TextPreprocessor                                │
│  1. Clean text (URLs, mentions, special chars)               │
│  2. Normalize slang ("gk" → "tidak")                         │
│  3. Remove stopwords (optional)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ Cleaned text (DataFrame)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          src/core/sentiment_analyzer.py                      │
│         SentimentAnalyzer (Hugging Face)                     │
│  Model: w11wo/indonesian-roberta-base-sentiment-classifier   │
└──────────────────────┬──────────────────────────────────────┘
                       │ DataFrame with `sentiment` column
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            src/core/visualizer.py                            │
│              SentimentVisualizer                             │
│  - Sentiment Distribution (Matplotlib)                       │
│  - Positive Word Cloud                                      │
│  - Negative Word Cloud                                      │
│  - Neutral Word Cloud                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ Images + DataFrame
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Gradio Output Panel                             │
│  Status │ Charts │ Data Table │ Excel Download │ JSON DL   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack

| Component | Library / Tool | Version | Role |
|---|---|---|---|
| Web Framework | Gradio | ^6.11.0 | Interactive UI for scraping + visualization |
| Scraping | Apify + apify-client | ^1.9.2 | Cloud actors for TikTok, Instagram, FB, X |
| NLP / ML | Hugging Face Transformers | ^4.51.0 | Pipeline for Indonesian sentiment RoBERTa |
| Sentiment Model | w11wo/indonesian-roberta-base-sentiment-classifier | — | Pre-trained Indonesian sentiment classifier |
| Data Processing | pandas | ^2.2.3 | DataFrame manipulation across the pipeline |
| Visualization | matplotlib + wordcloud | ^3.10.1 / ^1.9.4 | Distribution plots and word clouds |
| Environment | python-dotenv | ^1.0.1 | Load API tokens from `.env` |
| Package Manager | uv | ^0.6.7 | Fast dependency resolution and virtual env |
| Platform | Python | 3.14 (Space) / 3.10+ (local) | Runtime |

**Avoided alternatives (and why):**
- **Scrapy / Playwright** — Chosen Apify actors instead to avoid dealing with platform anti-bot measures, CAPTCHAs, and rate limits.
- **FastAPI + React** — Gradio is sufficient for a single-user analytical tool; no need for a separate frontend build step.
- **SQLite / PostgreSQL** — File-based JSON and Excel outputs are enough for the current scope; adding a DB would increase deployment complexity without clear benefit.
- **spaCy / NLTK** — The HF Transformers pipeline handles tokenization internally; custom Indonesian preprocessing is done with regex + dictionary lookups.

---

## 5. Project Structure

```
SentimentNusa/
├── app.py                          # Gradio application entry point
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (gitignored)
├── .env.example                    # Template for env vars
├── pyproject.toml                  # uv project metadata
├── uv.lock                         # Locked dependency versions
├── README.md                       # User-facing documentation
├── ARCHITECTURE.md                 # This document
├── models/                         # Hugging Face cache (local models)
├── data/
│   └── raw/                        # Timestamped raw JSON extraction results
├── src/
│   ├── config/
│   │   └── settings.py             # Constants, actor IDs, normalization dicts
│   ├── core/
│   │   ├── scraper.py              # SocialMediaScraper (Apify integration)
│   │   ├── preprocessor.py         # TextPreprocessor (clean + normalize)
│   │   ├── sentiment_analyzer.py   # SentimentAnalyzer (HF pipeline)
│   │   └── visualizer.py           # SentimentVisualizer (plots + word clouds)
│   └── utils/
│       └── helpers.py              # URL validation, Excel/JSON export, sample data
└── apify_endpoint/
    └── x.py                        # Reference script for X search actor input
```

---

## 6. Scraping Layer (`src/core/scraper.py`)

`SocialMediaScraper` is a thin wrapper around `apify_client.ApifyClient`. Each platform has a dedicated method that:
1. Constructs the actor-specific `run_input` dictionary
2. Calls `client.actor(actor_id).call(run_input=...)`
3. Iterates over the default dataset and returns a list of dicts

**Platform-specific actor configurations:**

| Platform | Actor ID | Input Keys | Output Schema Notes |
|---|---|---|---|
| TikTok | `BDec00yAmCm1QbMEI` | `postURLs`, `commentsPerPost`, `maxRepliesPerComment` | Comments nested under posts |
| Instagram | `SbK00X0JYCPblD2wp` | `directUrls`, `resultsLimit` | Flat list of comments |
| Facebook | `us5srxAYnsrkgUv2v` | `startUrls`, `resultsLimit`, `includeNestedComments` | Comments with reply nesting |
| X | `CJdippxWmn9uRfooo` | `searchTerms`, `maxItems`, `lang`, `filter:*` flags | Tweets matching search keywords |

**X actor specifics:**
- Unlike the other platforms, X does **not** take URLs; it takes **search keywords**.
- All `filter:*` booleans are default `False` (no filtering applied).
- Language is pinned to `in` (Indonesian) by default.
- The `comments_per_post` slider in the UI maps directly to `maxItems`.

---

## 7. Text Preprocessing Pipeline (`src/core/preprocessor.py`)

`TextPreprocessor` applies a deterministic, ordered pipeline:

```
Raw Text
  │
  ▼
┌─────────────────────┐
│ 1. clean_text()     │  → lowercase, remove URLs, @mentions, # symbols,
│                      │    strip non-alphabetic characters
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ 2. remove_repeated_ │  → "bagusss" → "bagus"
│    chars()          │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ 3. normalize_text() │  → slang dictionary lookup
│                      │    "gk" → "tidak", "bgt" → "banget"
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ 4. remove_stopwords │  → optional; removes common Indonesian stopwords
│    (optional)       │
└─────────────────────┘
```

**DataFrame integration:** `preprocess_dataframe()` drops empty/duplicate rows, applies the pipeline, and produces a `clean_text` column. The source column (e.g., `videoWebUrl` for TikTok) is preserved for provenance.

---

## 8. Sentiment Analysis (`src/core/sentiment_analyzer.py`)

`SentimentAnalyzer` wraps Hugging Face's `pipeline("sentiment-analysis", ...)`.

- **Model:** `w11wo/indonesian-roberta-base-sentiment-classifier`
- **Tokenizer max length:** 512 characters (truncated before inference)
- **Label normalization:** Maps `positive`/`pos`/`positif` → `positive`, `negative`/`neg`/`negatif` → `negative`, everything else → `neutral`
- **Failure mode:** Any error during inference returns `"neutral"` — the pipeline never crashes the application.

**Caching strategy:** The model is downloaded to `./models/` (not the system HF cache) so Hugging Face Spaces deployments are self-contained and fast on restart.

---

## 9. Visualization Layer (`src/core/visualizer.py`)

`SentimentVisualizer` generates:
1. **Sentiment Distribution** — Bar chart (matplotlib) of positive / negative / neutral counts
2. **Word Clouds** — Three separate word clouds (positive, negative, neutral) using the `wordcloud` library, filtered by sentiment label

All figures are returned as base64-encoded images compatible with Gradio's `gr.Image` component.

---

## 10. Configuration (`src/config/settings.py`)

All tunables are centralized here:

| Setting | Default | Description |
|---|---|---|
| `APIFY_TOKEN` | `""` (from `.env`) | Apify API token |
| `APIFY_ACTORS` | dict per platform | Actor IDs; X defaults to `CJdippxWmn9uRfooo` |
| `DEFAULT_COMMENTS_PER_POST` | `10` | Comments to fetch per URL |
| `DEFAULT_X_MAX_ITEMS` | `100` | Tweets to fetch for X search |
| `DEFAULT_X_SEARCH_LANG` | `"in"` | Language filter for X search |
| `SENTIMENT_MODEL` | `w11wo/indonesian-roberta-base-sentiment-classifier` | HF model name |
| `MAX_TEXT_LENGTH` | `512` | Truncation limit for model input |
| `NORMALIZATION_DICT` | ~30 entries | Slang → standard Indonesian mappings |
| `INDONESIAN_STOPWORDS` | ~40 entries | Common words to optionally strip |

---

## 11. Data Model

The application uses **pandas DataFrames** as the primary intermediate format. No formal schema is enforced, but the expected columns after each stage are:

| Stage | Key Columns | Type |
|---|---|---|
| After scraping | `text`, `source` (added by `merge_platform_data`) | `str`, `str` |
| After preprocessing | `text`, `clean_text`, `source` | `str`, `str`, `str` |
| After sentiment | `text`, `clean_text`, `sentiment`, `source` | `str`, `str`, `str`, `str` |

Raw API data is also kept as a list of dictionaries (the original Apify output) for JSON export.

---

## 12. Authentication & Secrets

- **Apify token** — Stored in `.env` as `APIFY_TOKEN`. Can also be entered dynamically in the Gradio UI sidebar (overrides `.env`).
- **HF token** — Optional; set as `HF_TOKEN` in `.env` only if hitting Hugging Face rate limits during model download.
- **No persistent user sessions** — The Gradio app is stateless per tab; tokens are not logged or stored server-side.

---

## 13. Deployment

### Local Development
```bash
uv venv
uv pip install -r requirements.txt
uv run app.py
# → http://localhost:7860
```

### Hugging Face Spaces
- **SDK:** Gradio
- **App file:** `app.py`
- **Python version:** 3.14 (as configured in README metadata)
- **Secrets:** `APIFY_TOKEN` and `HF_TOKEN` added via Space Settings → Secrets
- **Persistent storage:** `data/raw/` and `models/` directories are preserved between restarts

---

## 14. Implementation Status

| Phase | Status | Description |
|---|---|---|
| Project scaffolding | ✅ Done | uv project, folder structure, `.env` support |
| TikTok scraping | ✅ Done | Comments scraper integrated |
| Instagram scraping | ✅ Done | Comments scraper integrated |
| Facebook scraping | ✅ Done | Comments scraper integrated |
| X scraping | ✅ Done | Search-based actor (replaced URL-based reply scraper) |
| Text preprocessing | ✅ Done | Cleaning, slang normalization, stopword removal |
| Sentiment analysis | ✅ Done | Indonesian RoBERTa classifier |
| Visualizations | ✅ Done | Distribution chart + 3 word clouds |
| Excel export | ✅ Done | Full DataFrame export via `save_df_to_temp_excel` |
| JSON export | ✅ Done | Raw API data export via `save_raw_data_to_temp_json` |
| Sample data | ✅ Done | 25-sample dataset for token-free testing |
| Gradio UI | ✅ Done | Sidebar layout with platform selector and controls |
| Documentation | ✅ Done | README + ARCHITECTURE.md |

---

## 15. Decision Log

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Scraping approach | Apify cloud actors | Scrapy, Playwright, direct API | Avoids bot detection, CAPTCHA, and rate-limit complexity |
| Web framework | Gradio | FastAPI + React, Streamlit | Single-file app, zero frontend build, native Hugging Face Spaces support |
| Sentiment model | w11wo/indonesian-roberta-base | Multilingual BERT, VADER | Fine-tuned specifically for Indonesian social media; higher accuracy |
| Data storage | File-based (JSON + Excel) | SQLite, PostgreSQL | Simpler deployment, sufficient for single-user analytics |
| X workflow | Search keywords | Tweet URLs + reply scraping | New actor (`CJdippxWmn9uRfooo`) supports broader topic analysis instead of per-post replies |
| Model caching | Local `./models/` directory | System HF cache (`~/.cache`) | Self-contained for HF Spaces; faster cold starts |
| Package manager | uv | pip + venv, poetry | Faster installs, lockfile support, modern Python tooling |

---

## 16. Future Work

- **Multi-language support** — English or Javanese preprocessing pipelines (requires new models and normalization dictionaries)
- **Custom model fine-tuning** — Fine-tune the Indonesian RoBERTa on domain-specific data (e.g., political or product review text)
- **Database backend** — SQLite or PostgreSQL for historical analysis tracking and user sessions
- **Scheduled scraping** — Periodic background scraping with cron or Celery (requires moving beyond Gradio's synchronous model)
- **Advanced filters** — Date-range filtering, engagement threshold filtering, or geolocation filtering for X search
- **REST API** — Expose the scraping + analysis pipeline as a FastAPI endpoint for programmatic access
- **YouTube integration** — Add YouTube comments scraper via Apify actor

---

## 17. Known Limitations

1. **Synchronous scraping** — Apify actors block the Gradio UI until complete. Large `maxItems` values on X can cause timeouts.
2. **Approximate comment counts** — Some platforms (especially Facebook) return fewer comments than requested due to privacy or deletion.
3. **No deduplication across sessions** — Running the same analysis twice produces two separate raw JSON files.
4. **Model size** — The RoBERTa model is ~500MB; first download can be slow on low-bandwidth connections.
5. **X search language lock** — Currently hardcoded to Indonesian (`"in"`); changing it requires editing `settings.py`.
