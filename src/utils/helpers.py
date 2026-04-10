"""Utility helper functions for SentimentNusa."""
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import os


def validate_url(url: str, platform: str) -> bool:
    """Validate if a URL is valid for a specific platform.
    
    Args:
        url: URL to validate
        platform: Platform name ('tiktok' or 'instagram')
        
    Returns:
        True if valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.lower().strip()
    
    if platform == "tiktok":
        return "tiktok.com" in url or "tiktok" in url
    elif platform == "instagram":
        return "instagram.com" in url or "instagr.am" in url
    
    return False


def save_to_excel(
    df: pd.DataFrame,
    filename: Optional[str] = None,
    output_dir: str = "data",
) -> str:
    """Save DataFrame to Excel file.
    
    Args:
        df: DataFrame to save
        filename: Output filename (if None, generates timestamped name)
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentiment_analysis_{timestamp}.xlsx"
    
    filepath = os.path.join(output_dir, filename)
    df.to_excel(filepath, index=False)
    
    return filepath


def format_sentiment_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Format sentiment analysis results for display.
    
    Args:
        df: DataFrame with sentiment column
        
    Returns:
        Dictionary with formatted results
    """
    if "sentiment" not in df.columns:
        return {}
    
    counts = df["sentiment"].value_counts().to_dict()
    total = len(df)
    
    percentages = {
        sentiment: round((count / total) * 100, 2)
        for sentiment, count in counts.items()
    }
    
    return {
        "total_comments": total,
        "sentiment_counts": counts,
        "sentiment_percentages": percentages,
        "dominant_sentiment": max(counts, key=counts.get) if counts else None,
    }


def merge_platform_data(data_dict: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Merge data from multiple platforms into a single DataFrame.
    
    Args:
        data_dict: Dictionary with platform names as keys and lists of items as values
        
    Returns:
        Merged DataFrame with source column
    """
    all_data = []
    
    for platform, items in data_dict.items():
        for item in items:
            item["source"] = platform
            all_data.append(item)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.DataFrame(all_data)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."


def get_platform_from_url(url: str) -> Optional[str]:
    """Detect platform from URL.
    
    Args:
        url: URL to analyze
        
    Returns:
        Platform name or None
    """
    url = url.lower()
    
    if "tiktok.com" in url:
        return "tiktok"
    elif "instagram.com" in url or "instagr.am" in url:
        return "instagram"
    
    return None


def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing purposes with 25 diverse examples.
    
    Returns:
        Sample DataFrame with positive, negative, and neutral sentiments
    """
    sample_data = {
        "text": [
            # POSITIVE (10 examples)
            "Bagus banget videonya! Sangat membantu untuk belajar",
            "Saya suka konten ini, terus berkarya ya min",
            "Wow keren banget! Langsung subscribe channel ini",
            "Informasinya sangat bermanfaat, thanks for sharing!",
            "Kontennya keren dan informatif, suka deh",
            "Penjelasannya mudah dipahami, makasih banyak ya",
            "Video ini sangat inspiratif, jadi termotivasi",
            "Top markotop! Sangat recommended untuk ditonton",
            "Sangat edukatif dan menambah wawasan, thank you",
            "Bagus penjelasannya, jadi paham sekarang",
            
            # NEGATIVE (8 examples)
            "Kurang suka dengan videonya, tidak informatif sama sekali",
            "Jelek banget, tidak recommended buat ditonton",
            "Waste of time, kontennya tidak bermutu",
            "Kecewa sama hasilnya, ekspektasi tinggi realitanya zonk",
            "Gak jelas penjelasannya, bikin bingung aja",
            "Kualitas video buruk, audionya juga gak jelas",
            "Clickbait! Judulnya menarik tapi isinya gak sesuai",
            "Mending cari konten lain aja, ini gak worth it",
            
            # NEUTRAL (7 examples)
            "Biasa saja, tidak ada yang spesial dari video ini",
            "Standar, seperti konten lainnya di platform ini",
            "Lumayan oke untuk pemula yang mau belajar",
            "Not bad, tapi masih bisa lebih baik lagi",
            "Biasa aja sih, gak terlalu suka tapi gak benci juga",
            "Cukup informatif untuk level dasar",
            "Sesuai ekspektasi untuk konten semacam ini",
        ],
        "source": [
            # Sources corresponding to texts above
            "tiktok", "tiktok", "instagram", "instagram", "tiktok",
            "tiktok", "instagram", "instagram", "tiktok", "instagram",
            "instagram", "tiktok", "tiktok", "instagram", "instagram",
            "tiktok", "instagram", "tiktok",
            "instagram", "tiktok", "tiktok", "instagram", "tiktok",
            "tiktok", "instagram"
        ],
    }
    
    return pd.DataFrame(sample_data)
