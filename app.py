"""Main application entry point for SentimentNusa.

SentimentNusa is a sentiment analysis application that scrapes comments
from TikTok, Instagram, or Facebook and analyzes their sentiment.
"""
import os
import sys
from pathlib import Path

# CRITICAL: Set Hugging Face cache directory BEFORE importing transformers
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR / "hub")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.core.scraper import SocialMediaScraper
from src.core.preprocessor import TextPreprocessor
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.visualizer import SentimentVisualizer
from src.utils.helpers import (
    validate_url,
    save_df_to_temp_excel,
    save_raw_data_to_temp_json,
    format_sentiment_results,
    merge_platform_data,
    get_platform_from_url,
    create_sample_data,
    extract_tweet_id,
)


def scrape_and_analyze(
    platform: str,
    urls: str,
    comments_per_post: int,
    apify_token: str,
    progress=gr.Progress(),
) -> Tuple[str, pd.DataFrame, pd.DataFrame, str, str, str, str, List[Dict]]:
    """Scrape data from social media and perform sentiment analysis.

    Args:
        platform: Selected platform ('TikTok', 'Instagram', 'Facebook', or 'X')
        urls: Comma-separated post URLs for the selected platform
        comments_per_post: Number of comments to extract per post
        apify_token: Apify API token for scraping
        progress: Gradio progress tracker

    Returns:
        Tuple of (status message, display DataFrame, full DataFrame, distribution chart,
                 positive wordcloud, negative wordcloud, neutral wordcloud, raw_api_data)
    """
    try:
        # Validate Apify token
        if not apify_token or not apify_token.strip():
            return (
                "❌ Error: Apify API token is required.\n\n"
                "Please enter your Apify API token in the sidebar.\n"
                "Get your free token at: https://console.apify.com/settings/integrations",
                pd.DataFrame(),
                pd.DataFrame(),
                None,
                None,
                None,
                None,
                []
            )

        # Initialize components
        progress(0.1, desc="Initializing scraper...")
        scraper = SocialMediaScraper(token=apify_token.strip())
        preprocessor = TextPreprocessor()
        analyzer = SentimentAnalyzer()
        visualizer = SentimentVisualizer()
        
        # Parse inputs
        url_list = [url.strip() for url in urls.split(",") if url.strip()]
        
        if not url_list:
            return "Error: No valid URLs provided.", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
        
        all_data = {}
        platform_key = platform.lower()
        
        # Scrape based on selected platform
        if platform_key == "tiktok":
            progress(0.3, desc="Scraping TikTok...")
            valid_urls = [url for url in url_list if validate_url(url, "tiktok")]
            if valid_urls:
                tiktok_data = scraper.scrape_tiktok(
                    post_urls=valid_urls,
                    comments_per_post=comments_per_post,
                )
                all_data["tiktok"] = tiktok_data
            else:
                return "Error: No valid TikTok URLs provided.", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
                
        elif platform_key == "instagram":
            progress(0.3, desc="Scraping Instagram...")
            valid_urls = [url for url in url_list if validate_url(url, "instagram")]
            if valid_urls:
                instagram_data = scraper.scrape_instagram(
                    post_urls=valid_urls,
                    comments_per_post=comments_per_post,
                )
                all_data["instagram"] = instagram_data
            else:
                return "Error: No valid Instagram URLs provided.", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
                
        elif platform_key == "facebook":
            progress(0.3, desc="Scraping Facebook...")
            valid_urls = [url for url in url_list if validate_url(url, "facebook")]
            if valid_urls:
                facebook_data = scraper.scrape_facebook(
                    post_urls=valid_urls,
                    comments_per_post=comments_per_post,
                )
                all_data["facebook"] = facebook_data
            else:
                return "Error: No valid Facebook URLs provided.", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
                
        elif platform_key == "x":
            progress(0.3, desc="Scraping X (Twitter)...")
            valid_urls = [url for url in url_list if validate_url(url, "x")]
            if valid_urls:
                # Extract tweet IDs from URLs
                tweet_ids = []
                for url in valid_urls:
                    tweet_id = extract_tweet_id(url)
                    if tweet_id:
                        tweet_ids.append(tweet_id)
                
                if tweet_ids:
                    x_data = scraper.scrape_x(
                        tweet_ids=tweet_ids,
                        max_pages=max(1, comments_per_post // 35),  # Approximate pages based on comments limit
                    )
                    all_data["x"] = x_data
                else:
                    return "Error: Could not extract tweet IDs from URLs.", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
            else:
                return "Error: No valid X URLs provided. URLs should be in format: https://x.com/username/status/1234567890", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
        
        if not all_data:
            return "Error: No data retrieved. Please check your URLs.", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
        
        # Merge data
        progress(0.6, desc="Processing data...")
        df = merge_platform_data(all_data)
        
        if df.empty:
            return "Error: No data retrieved. Please check your URLs.", pd.DataFrame(), pd.DataFrame(), None, None, None, None, []
        
        # Preprocess
        progress(0.75, desc="Preprocessing text...")
        df = preprocessor.preprocess_dataframe(df, source_column="source")
        
        # Analyze sentiment
        progress(0.85, desc="Analyzing sentiment...")
        df = analyzer.analyze_dataframe(df)
        
        # Generate visualizations
        progress(0.95, desc="Generating visualizations...")
        viz = visualizer.get_all_visualizations(df)
        
        # Format results
        results_summary = format_sentiment_results(df)
        status_msg = f"""
        ✅ Analysis Complete!
        
        📊 Total Comments: {results_summary['total_comments']}
        
        📈 Sentiment Distribution:
        • Positive: {results_summary['sentiment_counts'].get('positive', 0)} ({results_summary['sentiment_percentages'].get('positive', 0)}%)
        • Negative: {results_summary['sentiment_counts'].get('negative', 0)} ({results_summary['sentiment_percentages'].get('negative', 0)}%)
        • Neutral: {results_summary['sentiment_counts'].get('neutral', 0)} ({results_summary['sentiment_percentages'].get('neutral', 0)}%)
        
        🏆 Dominant Sentiment: {results_summary['dominant_sentiment'] or 'N/A'}
        """
        
        progress(1.0, desc="Done!")
        
        # Select available columns for display
        display_cols = ["text", "clean_text", "sentiment"]
        if "source" in df.columns:
            display_cols.append("source")
        
        # Flatten raw data from all platforms into a single list
        raw_api_data = []
        for platform_items in all_data.values():
            raw_api_data.extend(platform_items)
        
        return (
            status_msg,
            df[display_cols],
            df,  # Full DataFrame for Excel download
            viz.get("sentiment_distribution"),
            viz.get("wordcloud_positive"),
            viz.get("wordcloud_negative"),
            viz.get("wordcloud_neutral"),
            raw_api_data,  # Raw API data for JSON download
        )
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, pd.DataFrame(), pd.DataFrame(), None, None, None, None, []


def analyze_sample_data() -> Tuple[str, pd.DataFrame, pd.DataFrame, str, str, str, str, List[Dict]]:
    """Analyze sample data for demonstration.
    
    Returns:
        Same format as scrape_and_analyze
    """
    try:
        preprocessor = TextPreprocessor()
        analyzer = SentimentAnalyzer()
        visualizer = SentimentVisualizer()
        
        # Create sample data
        df = create_sample_data()
        df = preprocessor.preprocess_dataframe(df, source_column="source")
        df = analyzer.analyze_dataframe(df)
        
        viz = visualizer.get_all_visualizations(df)
        
        results_summary = format_sentiment_results(df)
        status_msg = f"""
        ✅ Sample Analysis Complete!
        
        📊 Total Comments: {results_summary['total_comments']}
        
        📈 Sentiment Distribution:
        • Positive: {results_summary['sentiment_counts'].get('positive', 0)} ({results_summary['sentiment_percentages'].get('positive', 0)}%)
        • Negative: {results_summary['sentiment_counts'].get('negative', 0)} ({results_summary['sentiment_percentages'].get('negative', 0)}%)
        • Neutral: {results_summary['sentiment_counts'].get('neutral', 0)} ({results_summary['sentiment_percentages'].get('neutral', 0)}%)
        
        🏆 Dominant Sentiment: {results_summary['dominant_sentiment'] or 'N/A'}
        
        💡 Note: This is sample data. Use the "Scrape & Analyze" tab to analyze real social media posts.
        """
        
        # Select available columns for display
        display_cols = ["text", "clean_text", "sentiment"]
        if "source" in df.columns:
            display_cols.append("source")
        
        # Sample data has no raw API data
        raw_api_data = []
        
        return (
            status_msg,
            df[display_cols],
            df,  # Full DataFrame for Excel download
            viz.get("sentiment_distribution"),
            viz.get("wordcloud_positive"),
            viz.get("wordcloud_negative"),
            viz.get("wordcloud_neutral"),
            raw_api_data,  # Empty for sample data
        )
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, pd.DataFrame(), pd.DataFrame(), None, None, None, None, []


def download_excel(df: pd.DataFrame):
    """Prepare DataFrame for Excel download.
    
    Args:
        df: DataFrame to download
        
    Returns:
        Path to temporary file for Gradio File component
    """
    if df.empty:
        return None
    
    filepath = save_df_to_temp_excel(df)
    return filepath


def download_json(raw_data: List[Dict]):
    """Prepare raw API data for JSON download.
    
    Args:
        raw_data: Raw API data from scraper
        
    Returns:
        Path to temporary file for Gradio File component
    """
    if not raw_data:
        return None
    
    filepath = save_raw_data_to_temp_json(raw_data)
    return filepath


# Create Gradio Interface
def create_interface() -> gr.Blocks:
    """Create the Gradio interface with sidebar layout.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="SentimentNusa - Social Media Sentiment Analysis") as app:
        # Header
        gr.Markdown("""
        # 🎭 SentimentNusa
        ## Social Media Sentiment Analysis Tool
        """)
        
        # About Accordion
        with gr.Accordion(label="ℹ️ About SentimentNusa", open=False):
            gr.Markdown("""
            **SentimentNusa** is an Indonesian-focused sentiment analysis tool that helps you understand
            public opinion from social media comments.
            
            ### Features
            
            - 🔗 Multi-platform support (TikTok, Instagram, Facebook, X)
            - 🧹 Automatic text preprocessing and normalization
            - 🎯 Indonesian language sentiment analysis
            - 📊 Rich visualizations and word clouds
            - 💾 Export results to Excel and JSON
            
            ### How to Use

            1. **Setup**: Enter your Apify API token in the sidebar (get one free at [apify.com](https://console.apify.com/settings/integrations))
            2. **Select**: Choose the platform (TikTok, Instagram, Facebook, or X)
            3. **Input**: Enter post URLs for the selected platform (see supported formats below)
            4. **Analyze**: Click "Start Analysis" and wait for results
            5. **Download**: Download results as Excel (.xlsx) or JSON (.json) files
            
            ### 📎 Multiple URLs Input
            
            You can analyze multiple posts at once by entering multiple URLs separated by commas:
            ```
            https://www.tiktok.com/@user1/video/123, https://www.tiktok.com/@user2/video/456
            ```
            Or place each URL on a new line in the text box.
            
            ### Technology Stack
            
            - **Gradio**: Web interface
            - **Apify**: Social media scraping
            - **Hugging Face Transformers**: Sentiment analysis
            - **Indonesian RoBERTa**: Pre-trained sentiment model
            
            ### 🔗 Supported Formats:
            
            - **TikTok**: `https://www.tiktok.com/@username/video/1234567890`
            - **Instagram**: `https://www.instagram.com/p/ABC123DEF/` or `https://www.instagram.com/reel/ABC123DEF/`
            - **Facebook**: `https://www.facebook.com/page/posts/post-id` or `https://fb.com/...`
            - **X (Twitter)**: `https://x.com/username/status/1234567890` or `https://twitter.com/username/status/1234567890`
            """)
        
        # Sidebar with controls
        with gr.Sidebar(label="Controls", open=True):
            gr.Markdown("### API Configuration")

            apify_token_input = gr.Textbox(
                label="Apify API Token",
                placeholder="apify_api_xxxxxxxxxxxx",
                type="password",
                lines=1,
                info="Your personal Apify API token for scraping",
            )

            with gr.Accordion(label="🔑 How to get your Apify token", open=False):
                gr.Markdown("""
                ### Getting Your Apify API Token

                1. **Create a free Apify account** at [https://console.apify.com/sign-up](https://console.apify.com/sign-up)
                2. **Go to Settings** → [Integrations](https://console.apify.com/settings/integrations)
                3. **Copy your API token** (starts with `apify_api_`)
                4. **Paste it above** and start analyzing!

                ### Why do I need this?

                - Apify provides the scraping infrastructure for social media data
                - Each user needs their own token for security and rate limits
                - Your token is **never stored** - it only exists during your session

                ### Free Tier Limits

                - Apify offers a **free tier** with $5 credit monthly
                - This is enough for hundreds of comments
                - Perfect for testing and small projects
                """)

            gr.Markdown("---")
            gr.Markdown("### Select Platform & Input URLs")

            platform_selector = gr.Radio(
                label="Select Platform",
                choices=["TikTok", "Instagram", "Facebook", "X"],
                value="TikTok",
            )

            urls_input = gr.Textbox(
                label="Post URLs",
                placeholder="https://www.tiktok.com/@username/video/1234567890 or https://x.com/username/status/1234567890",
                lines=3,
            )

            comments_count = gr.Slider(
                label="Comments per Post",
                minimum=10,
                maximum=500,
                value=10,
                step=10,
            )

            analyze_btn = gr.Button("🚀 Start Analysis", variant="primary", size="lg")

            # Add spacer to push sample button to bottom
            gr.Markdown("<div style='flex-grow: 1;'></div>")

            # Sample data button at bottom of sidebar
            gr.Markdown("---")
            gr.Markdown("### Quick Test")
            gr.Markdown("Try with sample data without needing API tokens.")
            sample_btn = gr.Button("🎯 Try Sample Data", variant="secondary")
        
        # Main content area
        with gr.Column():
            status_output = gr.Textbox(
                label="Status",
                value="Ready to analyze. Configure settings in the sidebar and click 'Start Analysis'.",
                lines=10,
            )
            
            with gr.Row():
                dist_plot = gr.Image(label="📊 Sentiment Distribution")
            
            with gr.Row():
                wc_positive = gr.Image(label="😊 Positive Words")
                wc_negative = gr.Image(label="😞 Negative Words")
                wc_neutral = gr.Image(label="😐 Neutral Words")
            
            results_table = gr.DataFrame(
                label="📋 Detailed Results",
                interactive=False,
            )
            
            # Hidden states for downloads
            full_data_state = gr.State(value=pd.DataFrame())
            raw_data_state = gr.State(value=[])
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📥 Download Results")
                    with gr.Row():
                        excel_download = gr.File(
                            label="Excel (.xlsx)",
                            interactive=False,
                        )
                        json_download = gr.File(
                            label="JSON - Raw Data (.json)",
                            interactive=False,
                        )
        
        # Event handlers
        analyze_btn.click(
            fn=scrape_and_analyze,
            inputs=[platform_selector, urls_input, comments_count, apify_token_input],
            outputs=[status_output, results_table, full_data_state, dist_plot, wc_positive, wc_negative, wc_neutral, raw_data_state],
        )
        
        sample_btn.click(
            fn=analyze_sample_data,
            inputs=[],
            outputs=[status_output, results_table, full_data_state, dist_plot, wc_positive, wc_negative, wc_neutral, raw_data_state],
        )
        
        # Update download files when data changes
        full_data_state.change(
            fn=download_excel,
            inputs=[full_data_state],
            outputs=[excel_download],
        )
        
        raw_data_state.change(
            fn=download_json,
            inputs=[raw_data_state],
            outputs=[json_download],
        )
        
        # Footer info
        gr.Markdown("""
        ---
        💡 **Tip**: Enter your Apify API token in the sidebar to start scraping real data. Don't have one? [Get a free token here](https://console.apify.com/settings/integrations).

        **Note**: Each platform uses different Apify actors with specific configurations. Select the platform first, then enter the appropriate URLs.

        **X (Twitter) Note**: For X posts, the system extracts the tweet ID from the URL and scrapes replies/comments. Comments per post setting is approximate (1 page ≈ 35 comments).

        **Privacy**: Your API token is never stored or logged. It only exists during your session.
        """)
    
    return app


if __name__ == "__main__":
    # Create and launch the app
    app = create_interface()
    
    # Launch with public URL disabled for security
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
