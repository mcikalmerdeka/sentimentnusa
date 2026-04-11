---
title: SentimentNusa
emoji: 🎭
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.11.0"
python_version: "3.14"
app_file: app.py
pinned: false
---

![Deploy to Hugging Face](https://github.com/mcikalmerdeka/sentimentnusa/actions/workflows/deploy.yml/badge.svg)

# SentimentNusa

**SentimentNusa** is a social media sentiment analysis tool designed for Indonesian language content. It scrapes comments from TikTok and Instagram posts, analyzes sentiment using state-of-the-art NLP models, and provides rich visualizations of the results.

Try the live app on [Hugging Face Space](https://huggingface.co/spaces/mcikalmerdeka/sentiment-nusa)

Python
Gradio
License

## Features

- **Multi-Platform Support**: Analyze comments from TikTok and Instagram
- **Indonesian Language Optimized**: Uses `w11wo/indonesian-roberta-base-sentiment-classifier` for accurate sentiment detection
- **Smart Text Preprocessing**: Automatic cleaning, normalization, and slang translation
- **Raw Data Persistence**: Automatically saves raw extraction data for future reference
- **Rich Visualizations**: Sentiment distribution charts and word clouds
- **User-Friendly Interface**: Interactive Gradio web UI
- **Data Export**: Export results to Excel for further analysis

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [Apify API Token](https://console.apify.com/account/integrations) (for social media scraping)

### Installation

1. **Clone the repository**
  ```bash
   git clone https://github.com/mcikalmerdeka/sentimentnusa.git
   cd sentimentnusa
  ```
2. **Set up virtual environment**
  ```bash
   # Using uv (recommended)
   uv venv

   # Or using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  ```
3. **Install dependencies**
  ```bash
   # Using uv
   uv pip install -r requirements.txt

   # Or using pip
   pip install -r requirements.txt
  ```
4. **Configure environment variables**
  Copy `.env.example` to `.env` and add your Apify token:
   Edit the `.env` file:
   Get your token from: [https://console.apify.com/account/integrations](https://console.apify.com/account/integrations)
5. **Run the application**
  ```bash
   uv run app.py
   # Or: python app.py
  ```
6. **Open in browser**
  Navigate to: [http://localhost:7860](http://localhost:7860)

## Usage Guide

### Scraping and Analyzing Real Data

1. **Select Platform**: Choose either TikTok or Instagram from the platform selector
2. **Enter Post URLs**: Input one or more post URLs (comma-separated)
  - **TikTok**: `https://www.tiktok.com/@username/video/1234567890`
  - **Instagram**: `https://www.instagram.com/p/ABC123DEF/` or `https://www.instagram.com/reel/ABC123DEF/`
3. **Set Comments Limit**: Adjust the slider for number of comments to extract per post (10-500)
4. **Start Analysis**: Click the "Start Analysis" button
5. **View Results**: Review sentiment distribution, word clouds, and detailed data table
6. **Export Data**: Download results to Excel using the "Download Results" button

### Try Sample Data

Use the "Try Sample" tab to test the system with pre-loaded sample comments covering positive, negative, and neutral sentiments.

## Project Structure

```
SentimentNusa/
├── app.py                      # Main Gradio application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API tokens)
├── .env.example               # Example environment file
├── .opencode.json             # MCP server configuration
├── models/                     # Downloaded Hugging Face models (~500MB)
├── data/                       # Data directory
│   └── raw/                   # Raw extraction data (JSON files)
├── src/
│   ├── config/
│   │   └── settings.py        # Configuration and constants
│   ├── core/
│   │   ├── scraper.py         # Apify integration for social media scraping
│   │   ├── preprocessor.py    # Text cleaning and normalization
│   │   ├── sentiment_analyzer.py  # Sentiment analysis using transformers
│   │   └── visualizer.py      # Charts and word clouds
│   └── utils/
│       └── helpers.py         # Utility functions
└── reference/                  # Reference documentation
```

## Technology Stack


| Component           | Technology                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Web Framework**   | [Gradio](https://gradio.app/)                                                                                                   |
| **Scraping**        | [Apify](https://apify.com/) + [apify-client](https://pypi.org/project/apify-client/)                                            |
| **ML/NLP**          | [Hugging Face Transformers](https://huggingface.co/docs/transformers/)                                                          |
| **Sentiment Model** | [w11wo/indonesian-roberta-base-sentiment-classifier](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier) |
| **Visualization**   | [Matplotlib](https://matplotlib.org/) + [WordCloud](https://github.com/amueller/word_cloud)                                     |
| **Data Processing** | [pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/)                                                              |


## Text Preprocessing Pipeline

1. **URL and Mention Removal**: Strips `http://...` and `@username`
2. **Case Normalization**: Converts to lowercase
3. **Special Character Removal**: Removes emojis and non-alphabetic characters
4. **Repeated Character Limiting**: Normalizes words like "bagusss" to "bagus"
5. **Slang Normalization**: Translates Indonesian slang:
  - `gk`, `ga`, `gak` -> `tidak`
  - `bgt`, `bgtt` -> `banget`
  - `dg`, `dgn` -> `dengan`
  - And more...
6. **Stopword Removal** (optional)

## Raw Data Storage

Every scraping operation automatically saves the raw JSON data to `data/raw/` with timestamped filenames:

- `tiktok_raw_20250410_143022.json`
- `instagram_raw_20250410_143025.json`

This provides a complete backup of the original Apify response data for future reference, debugging, or reprocessing.

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
APIFY_TOKEN=your_apify_token_here

# Apify Actor IDs (optional - uses defaults if not set)
APIFY_TIKTOK_ACTOR=BDec00yAmCm1QbMEI
APIFY_INSTAGRAM_ACTOR=SbK00X0JYCPblD2wp

# Optional
HUGGINGFACE_TOKEN=your_hf_token_here  # If you hit rate limits
```

### Apify Actors

The application uses the following public Apify actors:


| Platform  | Actor ID            | Description             |
| --------- | ------------------- | ----------------------- |
| TikTok    | `BDec00yAmCm1QbMEI` | TikTok Comments Scraper |
| Instagram | `SbK00X0JYCPblD2wp` | Instagram Scraper       |


These actor IDs can be customized via environment variables if needed.

## Example Output

### Sentiment Distribution

```
Analysis Complete!

Total Comments: 25

Sentiment Distribution:
- Positive: 10 (40.0%)
- Negative: 8 (32.0%)
- Neutral: 7 (28.0%)

Dominant Sentiment: positive
```

### Sample Analysis Results


| Original Text                 | Cleaned Text                   | Sentiment | Source    |
| ----------------------------- | ------------------------------ | --------- | --------- |
| Bagus banget videonya!        | bagus banget videonya          | positive  | tiktok    |
| Jelek banget, gak recommended | jelek banget tidak recommended | negative  | instagram |
| Biasa aja sih                 | biasa saja                     | neutral   | tiktok    |


## Troubleshooting

### Model Download Issues

If the model downloads to your system cache instead of the local `models/` folder:

1. Stop the application (Ctrl+C)
2. Delete from system cache: `C:\Users\<YourName>\.cache\huggingface\hub\`
3. Restart the application

### Apify Rate Limits

If you hit rate limits:

1. Wait a few minutes before retrying
2. Consider upgrading your Apify plan
3. Reduce the comments per post value

### Invalid Input Errors

Ensure your URLs match the expected format:

- TikTok URLs must contain `tiktok.com`
- Instagram URLs must contain `instagram.com` or `instagr.am`

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [w11wo](https://huggingface.co/w11wo) for the Indonesian RoBERTa sentiment model
- [Apify](https://apify.com/) for social media scraping infrastructure
- [Gradio](https://gradio.app/) for the intuitive web interface

## Contact

For questions or support, please open an issue on GitHub.

---

