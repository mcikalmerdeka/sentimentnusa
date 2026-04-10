"""Sentiment analysis module for analyzing text sentiment."""
import pandas as pd
from typing import List, Dict, Any, Optional
from transformers import pipeline
from src.config.settings import SENTIMENT_MODEL, MAX_TEXT_LENGTH, MODELS_DIR


class SentimentAnalyzer:
    """Analyzer for performing sentiment analysis on text data."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the Hugging Face model to use.
                       If None, uses the default Indonesian model.
        """
        self.model_name = model_name or SENTIMENT_MODEL
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model."""
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                cache_dir=str(MODELS_DIR),
                model_kwargs={"cache_dir": str(MODELS_DIR)},
            )
            print(f"Model {self.model_name} loaded successfully from {MODELS_DIR}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.classifier = None
    
    def analyze(self, text: str) -> str:
        """Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment label (positive, negative, or neutral)
        """
        if not text or not isinstance(text, str) or text.strip() == "":
            return "neutral"
        
        if self.classifier is None:
            return "neutral"
        
        try:
            # Truncate text if too long
            text = text[:MAX_TEXT_LENGTH]
            
            result = self.classifier(text)
            label = result[0]["label"].lower()
            
            # Normalize label format
            if label in ["positive", "pos", "positif"]:
                return "positive"
            elif label in ["negative", "neg", "negatif"]:
                return "negative"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return "neutral"
    
    def analyze_batch(self, texts: List[str]) -> List[str]:
        """Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment labels
        """
        return [self.analyze(text) for text in texts]
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_text",
    ) -> pd.DataFrame:
        """Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text to analyze
            
        Returns:
            DataFrame with added 'sentiment' column
        """
        df = df.copy()
        
        print("Analyzing sentiment, please wait...")
        df["sentiment"] = df[text_column].apply(self.analyze)
        
        print(f"Sentiment analysis complete. Found:")
        print(df["sentiment"].value_counts())
        
        return df
    
    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Get sentiment distribution from a DataFrame.
        
        Args:
            df: DataFrame with sentiment column
            
        Returns:
            Dictionary with sentiment counts
        """
        if "sentiment" not in df.columns:
            return {}
        
        return df["sentiment"].value_counts().to_dict()
