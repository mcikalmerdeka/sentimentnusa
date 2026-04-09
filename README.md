# 🎭 SentimentNusa

**SentimentNusa** is a comprehensive social media sentiment analysis tool that scrapes, analyzes, and visualizes public opinion from **TikTok**, **Instagram**, and **X (Twitter)** comments. Built specifically for Indonesian language content using state-of-the-art NLP models.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🔗 **Multi-Platform Support**: Analyze comments from TikTok, Instagram, and X (Twitter)
- 🇮🇩 **Indonesian Language Optimized**: Uses `w11wo/indonesian-roberta-base-sentiment-classifier` for accurate sentiment detection
- 🧹 **Smart Text Preprocessing**: Automatic cleaning, normalization, and slang translation
- 📊 **Rich Visualizations**: Sentiment distribution charts, platform breakdowns, and word clouds
- 🖥️ **User-Friendly Interface**: Interactive Gradio web UI for easy analysis
- 💾 **Data Export**: Export results to Excel for further analysis
- 🤖 **ML Training**: Train custom Naive Bayes and Random Forest classifiers

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [Apify API Token](https://console.apify.com/account/integrations) (for social media scraping)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/SentimentNusa.git
   cd SentimentNusa
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

4. **Configure API Token**

   Edit the `.env` file and add your Apify token:

   ```env
   APIFY_TOKEN=your_apify_token_here
   ```

   Get your token from: https://console.apify.com/account/integrations

5. **Run the application**

   ```bash
   uv run app.py
   # Or: python app.py
   ```

6. **Open in browser**

   Navigate to: http://localhost:7860

## 📖 Usage Guide

### 🔍 Scraping & Analyzing Real Data

1. **TikTok**: Enter post URLs

   ```
   https://www.tiktok.com/@username/video/1234567890
   ```

2. **Instagram**: Enter post URLs

   ```
   https://www.instagram.com/p/ABC123DEF/
   ```

3. **X (Twitter)**: Enter search keywords

   ```
   #indonesia, mbg, pemilu2024
   ```

4. Set the number of comments to extract per post
5. Click **"🚀 Start Analysis"**

### 📊 Try Sample Data

Use the **"📊 Try Sample"** tab to test the system with 25 pre-loaded sample comments covering positive, negative, and neutral sentiments across all platforms.

## 🏗️ Project Structure

```
SentimentNusa/
├── app.py                      # Main Gradio application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API tokens)
├── .opencode.json             # MCP server configuration
├── models/                     # Downloaded Hugging Face models (~500MB)
├── data/                       # Exported Excel files
├── src/
│   ├── config/
│   │   └── settings.py        # Configuration and constants
│   ├── core/
│   │   ├── scraper.py         # Apify integration for social media scraping
│   │   ├── preprocessor.py    # Text cleaning and normalization
│   │   ├── sentiment_analyzer.py  # Sentiment analysis using transformers
│   │   └── visualizer.py      # Charts, graphs, and word clouds
│   └── utils/
│       └── helpers.py         # Utility functions
└── reference/                  # Reference notebooks and documentation
```

## 🛠️ Technology Stack

| Component           | Technology                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Web Framework**   | [Gradio](https://gradio.app/)                                                                                                   |
| **Scraping**        | [Apify](https://apify.com/) + [apify-client](https://pypi.org/project/apify-client/)                                            |
| **ML/NLP**          | [Hugging Face Transformers](https://huggingface.co/docs/transformers/)                                                          |
| **Sentiment Model** | [w11wo/indonesian-roberta-base-sentiment-classifier](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier) |
| **ML Training**     | [scikit-learn](https://scikit-learn.org/)                                                                                       |
| **Visualization**   | [Matplotlib](https://matplotlib.org/) + [WordCloud](https://github.com/amueller/word_cloud)                                     |
| **Data Processing** | [pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/)                                                              |

## 🧹 Text Preprocessing Pipeline

1. **URL & Mention Removal**: Strips `http://...` and `@username`
2. **Case Normalization**: Converts to lowercase
3. **Special Character Removal**: Removes emojis and non-alphabetic characters
4. **Repeated Character Limiting**: Normalizes words like "bagusss" → "bagus"
5. **Slang Normalization**: Translates Indonesian slang:
   - `gk`, `ga`, `gak` → `tidak`
   - `bgt`, `bgtt` → `banget`
   - `dg`, `dgn` → `dengan`
   - And more...
6. **Stopword Removal** (optional)

## 📊 Output Visualizations

- **📈 Sentiment Distribution**: Bar chart showing counts of positive, negative, and neutral comments
- **☁️ Word Clouds**: Separate word clouds for each sentiment category
- **📱 Platform Breakdown**: Stacked bar chart showing sentiment by platform
- **📋 Detailed Table**: Full results with original text, cleaned text, sentiment, and source

## 🔧 Configuration

### Apify Actors Used

| Platform    | Actor ID                          | Description                |
| ----------- | --------------------------------- | -------------------------- |
| TikTok      | `BDec00yAmCm1QbMEI`               | TikTok Comments Scraper    |
| Instagram   | `apify/instagram-comment-scraper` | Instagram Comments Scraper |
| X (Twitter) | `apify/twitter-scraper`           | X Search Scraper           |

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
APIFY_TOKEN=your_apify_token_here

# Optional
HUGGINGFACE_TOKEN=your_hf_token_here  # If you hit rate limits
```

## 📝 Example Output

### Sentiment Distribution

```
📊 Total Comments: 25

📈 Sentiment Distribution:
• Positive: 10 (40.0%)
• Negative: 8 (32.0%)
• Neutral: 7 (28.0%)

🏆 Dominant Sentiment: positive
```

### Sample Analysis Results

| Original Text                 | Cleaned Text                   | Sentiment | Source    |
| ----------------------------- | ------------------------------ | --------- | --------- |
| Bagus banget videonya!        | bagus banget videonya          | positive  | tiktok    |
| Jelek banget, gak recommended | jelek banget tidak recommended | negative  | instagram |
| Biasa aja sih                 | biasa saja                     | neutral   | x         |

## 🐛 Troubleshooting

### Model Download Issues

If the model downloads to your system cache instead of the local `models/` folder:

1. Stop the application (Ctrl+C)
2. Delete from system cache: `C:\Users\<YourName>\.cache\huggingface\hub\`
3. Restart the application

### HTTP Warnings

The `Invalid HTTP request received` warnings are from the underlying Uvicorn server and don't affect functionality. They're typically caused by browser extensions or network scanning tools.

### Apify Rate Limits

If you hit rate limits:

1. Wait a few minutes before retrying
2. Consider upgrading your Apify plan
3. Reduce the `comments_per_post` value

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [w11wo](https://huggingface.co/w11wo) for the Indonesian RoBERTa sentiment model
- [Apify](https://apify.com/) for social media scraping infrastructure
- [Gradio](https://gradio.app/) for the intuitive web interface

## 📬 Contact

For questions or support, please open an issue on GitHub.

---
