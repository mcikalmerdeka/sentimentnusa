"""Text preprocessing module for cleaning and normalizing text data."""
import re
import pandas as pd
from typing import List, Optional
from src.config.settings import NORMALIZATION_DICT, INDONESIAN_STOPWORDS


class TextPreprocessor:
    """Preprocessor for cleaning and normalizing social media text."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.normalization_dict = NORMALIZATION_DICT
        self.stopwords = set(INDONESIAN_STOPWORDS)
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing URLs, mentions, special characters, etc.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Remove mentions (@username)
        text = re.sub(r"@\w+", "", text)
        
        # Remove hashtags symbol but keep the text
        text = re.sub(r"#", "", text)
        
        # Remove emojis and special characters (keep only letters and spaces)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def remove_repeated_chars(self, text: str) -> str:
        """Remove repeated characters (e.g., 'bagusss' -> 'bagus').
        
        Args:
            text: Text to process
            
        Returns:
            Text with repeated characters limited
        """
        # Limit to maximum 2 consecutive characters
        return re.sub(r"(.)\1+", r"\1\1", text)
    
    def normalize_text(self, text: str) -> str:
        """Normalize informal Indonesian words to standard form.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        words = text.split()
        normalized = [self.normalization_dict.get(word, word) for word in words]
        return " ".join(normalized)
    
    def remove_stopwords(self, text: str) -> str:
        """Remove common stopwords from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text without stopwords
        """
        words = text.split()
        filtered = [word for word in words if word not in self.stopwords]
        return " ".join(filtered)
    
    def preprocess(self, text: str, remove_stops: bool = False) -> str:
        """Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw text to preprocess
            remove_stops: Whether to remove stopwords
            
        Returns:
            Fully preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Apply cleaning steps
        text = self.clean_text(text)
        text = self.remove_repeated_chars(text)
        text = self.normalize_text(text)
        
        if remove_stops:
            text = self.remove_stopwords(text)
        
        return text
    
    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        source_column: str = "videoWebUrl",
        remove_stops: bool = False,
    ) -> pd.DataFrame:
        """Preprocess text data in a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            source_column: Name of the column containing source URL
            remove_stops: Whether to remove stopwords
            
        Returns:
            Preprocessed DataFrame with new 'clean_text' column
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Select only necessary columns if they exist
        available_cols = [col for col in [text_column, source_column] if col in df.columns]
        df = df[available_cols].copy()
        
        # Remove rows with empty text
        df = df.dropna(subset=[text_column])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[text_column])
        
        # Apply preprocessing
        df["clean_text"] = df[text_column].apply(
            lambda x: self.preprocess(x, remove_stops=remove_stops)
        )
        
        # Remove rows where cleaning resulted in empty text
        df = df[df["clean_text"].str.len() > 0]
        
        return df.reset_index(drop=True)
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text.
        
        Args:
            text: Text to extract hashtags from
            
        Returns:
            List of hashtags (without # symbol)
        """
        hashtags = re.findall(r"#(\w+)", text)
        return hashtags
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text.
        
        Args:
            text: Text to extract mentions from
            
        Returns:
            List of mentions (without @ symbol)
        """
        mentions = re.findall(r"@(\w+)", text)
        return mentions
