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
from typing import List, Optional, Tuple

from src.core.scraper import SocialMediaScraper
from src.core.preprocessor import TextPreprocessor
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.visualizer import SentimentVisualizer
from src.utils.helpers import (
    validate_url,
    save_to_excel,
    format_sentiment_results,
    merge_platform_data,
    get_platform_from_url,
    create_sample_data,
)


def scrape_and_analyze(
    platform: str,
    urls: str,
    comments_per_post: int,
    progress=gr.Progress(),
) -> Tuple[str, pd.DataFrame, str, str, str, str]:
    """Scrape data from social media and perform sentiment analysis.
    
    Args:
        platform: Selected platform ('TikTok', 'Instagram', or 'Facebook')
        urls: Comma-separated post URLs for the selected platform
        comments_per_post: Number of comments to extract per post
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (status message, results DataFrame, distribution chart,
                 positive wordcloud, negative wordcloud, neutral wordcloud)
    """
    try:
        # Initialize components
        progress(0.1, desc="Initializing scraper...")
        scraper = SocialMediaScraper()
        preprocessor = TextPreprocessor()
        analyzer = SentimentAnalyzer()
        visualizer = SentimentVisualizer()
        
        # Parse inputs
        url_list = [url.strip() for url in urls.split(",") if url.strip()]
        
        if not url_list:
            return "Error: No valid URLs provided.", pd.DataFrame(), None, None, None, None
        
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
                return "Error: No valid TikTok URLs provided.", pd.DataFrame(), None, None, None, None
                
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
                return "Error: No valid Instagram URLs provided.", pd.DataFrame(), None, None, None, None
                
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
                return "Error: No valid Facebook URLs provided.", pd.DataFrame(), None, None, None, None
        
        if not all_data:
            return "Error: No data retrieved. Please check your URLs.", pd.DataFrame(), None, None, None, None
        
        # Merge data
        progress(0.6, desc="Processing data...")
        df = merge_platform_data(all_data)
        
        if df.empty:
            return "Error: No data retrieved. Please check your URLs.", pd.DataFrame(), None, None, None, None
        
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
        
        return (
            status_msg,
            df[display_cols],
            viz.get("sentiment_distribution"),
            viz.get("wordcloud_positive"),
            viz.get("wordcloud_negative"),
            viz.get("wordcloud_neutral"),
        )
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, pd.DataFrame(), None, None, None, None


def analyze_sample_data() -> Tuple[str, pd.DataFrame, str, str, str, str]:
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
        
        return (
            status_msg,
            df[display_cols],
            viz.get("sentiment_distribution"),
            viz.get("wordcloud_positive"),
            viz.get("wordcloud_negative"),
            viz.get("wordcloud_neutral"),
        )
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, pd.DataFrame(), None, None, None, None


def download_results(df: pd.DataFrame) -> str:
    """Save results to Excel and return file path.
    
    Args:
        df: DataFrame to save
        
    Returns:
        Path to saved file
    """
    if df.empty:
        return "Error: No data to save"
    
    filepath = save_to_excel(df)
    return f"✅ Results saved to: {filepath}"


# Create Gradio Interface
def create_interface() -> gr.Blocks:
    """Create the Gradio interface.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="SentimentNusa - Social Media Sentiment Analysis") as app:
        gr.Markdown("""
        # 🎭 SentimentNusa
        ## Social Media Sentiment Analysis Tool

        Analyze sentiment from comments on **TikTok**, **Instagram**, or **Facebook**.

        🔗 **Supported Formats:**
        - **TikTok**: `https://www.tiktok.com/@username/video/1234567890`
        - **Instagram**: `https://www.instagram.com/p/ABC123DEF/` or `https://www.instagram.com/reel/ABC123DEF/`
        - **Facebook**: `https://www.facebook.com/page/posts/post-id` or `https://fb.com/...`
        """)
        
        with gr.Tabs():
            # Tab 1: Scrape & Analyze
            with gr.TabItem("🔍 Scrape & Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Select Platform & Input URLs")

                        platform_selector = gr.Radio(
                            label="Select Platform",
                            choices=["TikTok", "Instagram", "Facebook"],
                            value="TikTok",
                        )

                        urls_input = gr.Textbox(
                            label="Post URLs",
                            placeholder="https://www.tiktok.com/@username/video/1234567890",
                            lines=3,
                        )

                        comments_count = gr.Slider(
                            label="Comments per Post",
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                        )

                        analyze_btn = gr.Button("🚀 Start Analysis", variant="primary")
                    
                    with gr.Column(scale=2):
                        status_output = gr.Textbox(
                            label="Status",
                            value="Ready to analyze. Enter URLs/keywords and click 'Start Analysis'.",
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
                        
                        download_btn = gr.Button("💾 Download Results to Excel")
                        download_status = gr.Textbox(label="Download Status")
                
                # Event handlers
                analyze_btn.click(
                    fn=scrape_and_analyze,
                    inputs=[platform_selector, urls_input, comments_count],
                    outputs=[status_output, results_table, dist_plot, wc_positive, wc_negative, wc_neutral],
                )
                
                download_btn.click(
                    fn=download_results,
                    inputs=[results_table],
                    outputs=[download_status],
                )
            
            # Tab 2: Sample Analysis
            with gr.TabItem("📊 Try Sample"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### Try with Sample Data
                        
                        Click the button below to analyze sample comments.
                        This is useful for testing the system without needing API tokens.
                        """)
                        sample_btn = gr.Button("🎯 Analyze Sample Data", variant="secondary")
                    
                    with gr.Column(scale=2):
                        sample_status = gr.Textbox(label="Status", lines=10)
                        
                        with gr.Row():
                            sample_dist = gr.Image(label="📊 Sentiment Distribution")
                        
                        with gr.Row():
                            sample_wc_pos = gr.Image(label="😊 Positive Words")
                            sample_wc_neg = gr.Image(label="😞 Negative Words")
                            sample_wc_neu = gr.Image(label="😐 Neutral Words")
                        
                        sample_table = gr.DataFrame(label="📋 Sample Results")
                
                sample_btn.click(
                    fn=analyze_sample_data,
                    inputs=[],
                    outputs=[sample_status, sample_table, sample_dist, sample_wc_pos, sample_wc_neg, sample_wc_neu],
                )
            
            # Tab 3: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About SentimentNusa

                **SentimentNusa** is an Indonesian-focused sentiment analysis tool that helps you understand
                public opinion from social media comments.

                ### Features

                - 🔗 Multi-platform support (TikTok, Instagram, Facebook)
                - 🧹 Automatic text preprocessing and normalization
                - 🎯 Indonesian language sentiment analysis
                - 📊 Rich visualizations and word clouds
                - 💾 Export results to Excel

                ### How to Use

                1. **Setup**: Add your Apify API token to the `.env` file
                2. **Select**: Choose the platform (TikTok, Instagram, or Facebook)
                3. **Input**: Enter post URLs for the selected platform
                4. **Analyze**: Click "Start Analysis" and wait for results
                5. **Download**: Save your results to Excel for further analysis

                ### Technology Stack

                - **Gradio**: Web interface
                - **Apify**: Social media scraping
                - **Hugging Face Transformers**: Sentiment analysis
                - **Indonesian RoBERTa**: Pre-trained sentiment model

                """)
        
        gr.Markdown("""
        ---
        💡 **Tip**: Make sure to add your Apify API token to the `.env` file before scraping real data.

        **Note**: Each platform uses different Apify actors with specific configurations. Select the platform first, then enter the appropriate URLs.
        
        **Facebook Note**: The Facebook scraper uses the Apify Facebook Comments Scraper actor. You can configure the actor ID via the `APIFY_FACEBOOK_ACTOR` environment variable.
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
