"""Sentiment analysis module for analyzing text sentiment."""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


class MLClassifier:
    """Machine Learning classifier for sentiment analysis."""
    
    def __init__(self):
        """Initialize the ML classifier."""
        self.model_nb = None
        self.model_rf = None
        self.is_trained = False
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train ML classifiers on labeled data.
        
        Args:
            texts: List of texts
            labels: List of corresponding labels
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
        )
        
        # Train Naive Bayes
        self.model_nb = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ])
        self.model_nb.fit(X_train, y_train)
        y_pred_nb = self.model_nb.predict(X_test)
        
        # Train Random Forest
        self.model_rf = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ])
        self.model_rf.fit(X_train, y_train)
        y_pred_rf = self.model_rf.predict(X_test)
        
        # Calculate metrics
        accuracy_nb = accuracy_score(y_test, y_pred_nb)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        
        self.is_trained = True
        
        return {
            "naive_bayes_accuracy": accuracy_nb,
            "random_forest_accuracy": accuracy_rf,
            "naive_bayes_report": classification_report(y_test, y_pred_nb, output_dict=True),
            "random_forest_report": classification_report(y_test, y_pred_rf, output_dict=True),
            "confusion_matrix_nb": confusion_matrix(y_test, y_pred_nb).tolist(),
            "confusion_matrix_rf": confusion_matrix(y_test, y_pred_rf).tolist(),
        }
    
    def predict(self, texts: List[str], model: str = "naive_bayes") -> List[str]:
        """Predict sentiment using trained model.
        
        Args:
            texts: List of texts to predict
            model: Model to use ('naive_bayes' or 'random_forest')
            
        Returns:
            List of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if model == "naive_bayes":
            return self.model_nb.predict(texts).tolist()
        elif model == "random_forest":
            return self.model_rf.predict(texts).tolist()
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def predict_single(self, text: str, model: str = "naive_bayes") -> str:
        """Predict sentiment for a single text.
        
        Args:
            text: Text to predict
            model: Model to use ('naive_bayes' or 'random_forest')
            
        Returns:
            Predicted label
        """
        return self.predict([text], model)[0]
