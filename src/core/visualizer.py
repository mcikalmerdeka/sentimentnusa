"""Visualization module for creating charts and graphs."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from typing import Optional, List, Dict, Any, Union
import io
import base64
from PIL import Image
import numpy as np
from src.config.settings import (
    DEFAULT_FIGURE_SIZE,
    WORD_CLOUD_WIDTH,
    WORD_CLOUD_HEIGHT,
    INDONESIAN_STOPWORDS,
)


class SentimentVisualizer:
    """Visualizer for creating sentiment analysis visualizations."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.stopwords = set(INDONESIAN_STOPWORDS)
        self.stopwords.update(STOPWORDS)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = DEFAULT_FIGURE_SIZE
    
    def plot_sentiment_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Sentiment Analysis Distribution",
    ) -> Optional[Image.Image]:
        """Plot overall sentiment distribution as a bar chart.
        
        Args:
            df: DataFrame with sentiment column
            title: Chart title
            
        Returns:
            PIL Image object or None
        """
        if "sentiment" not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
        
        sentiment_counts = df["sentiment"].value_counts()
        colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}
        bar_colors = [colors.get(s, "#95a5a6") for s in sentiment_counts.index]
        
        sentiment_counts.plot(kind="bar", ax=ax, color=bar_colors)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Sentiment", fontsize=12)
        ax.set_ylabel("Number of Comments", fontsize=12)
        ax.tick_params(axis="x", rotation=0)
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def plot_sentiment_by_source(
        self,
        df: pd.DataFrame,
        source_column: str = "source",
        title: str = "Sentiment Analysis by Source",
    ) -> Optional[Image.Image]:
        """Plot sentiment distribution grouped by source.
        
        Args:
            df: DataFrame with sentiment and source columns
            source_column: Name of the source column
            title: Chart title
            
        Returns:
            PIL Image object or None
        """
        if "sentiment" not in df.columns or source_column not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create pivot table
        sentiment_by_source = df.groupby([source_column, "sentiment"]).size().unstack(fill_value=0)
        
        # Plot stacked bar
        sentiment_by_source.plot(kind="bar", stacked=True, ax=ax, 
                                 color=["#e74c3c", "#3498db", "#2ecc71"])
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Source", fontsize=12)
        ax.set_ylabel("Number of Comments", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Sentiment")
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def generate_wordcloud(
        self,
        df: pd.DataFrame,
        sentiment: str,
        text_column: str = "clean_text",
        colormap: str = "viridis",
    ) -> Optional[Image.Image]:
        """Generate word cloud for a specific sentiment.
        
        Args:
            df: DataFrame with sentiment and text columns
            sentiment: Sentiment to filter ('positive', 'negative', or 'neutral')
            text_column: Name of the text column
            colormap: Matplotlib colormap to use
            
        Returns:
            PIL Image object or None
        """
        if "sentiment" not in df.columns or text_column not in df.columns:
            return None
        
        # Filter data by sentiment
        filtered_df = df[df["sentiment"] == sentiment]
        
        if len(filtered_df) == 0:
            return None
        
        # Join all texts
        text = " ".join(filtered_df[text_column].astype(str))
        
        if not text.strip():
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=WORD_CLOUD_WIDTH,
            height=WORD_CLOUD_HEIGHT,
            background_color="white",
            colormap=colormap,
            stopwords=self.stopwords,
            max_words=100,
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        
        sentiment_titles = {
            "positive": "Word Cloud - POSITIVE Sentiment",
            "negative": "Word Cloud - NEGATIVE Sentiment",
            "neutral": "Word Cloud - NEUTRAL Sentiment",
        }
        ax.set_title(sentiment_titles.get(sentiment, f"Word Cloud - {sentiment.title()}"), 
                     fontsize=15)
        ax.axis("off")
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def plot_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        title: str = "Confusion Matrix",
    ) -> Optional[Image.Image]:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Chart title
            
        Returns:
            PIL Image object or None
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        
        plt.tight_layout()
        
        return self._fig_to_pil(fig)
    
    def get_all_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate all visualizations at once.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dictionary containing all visualization PIL Images
        """
        visualizations = {}
        
        # Overall sentiment distribution
        visualizations["sentiment_distribution"] = self.plot_sentiment_distribution(df)
        
        # Word clouds for each sentiment
        for sentiment in ["positive", "negative", "neutral"]:
            key = f"wordcloud_{sentiment}"
            colormaps = {
                "positive": "viridis",
                "negative": "Reds",
                "neutral": "cool",
            }
            visualizations[key] = self.generate_wordcloud(
                df, sentiment, colormap=colormaps[sentiment]
            )
        
        # Sentiment by source (if source column exists)
        if "source" in df.columns:
            visualizations["sentiment_by_source"] = self.plot_sentiment_by_source(df)
        
        return visualizations
    
    def _fig_to_pil(self, fig) -> Image.Image:
        """Convert matplotlib figure to PIL Image.
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            PIL Image object
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
