"""
sentiment_analysis.py
---------------------
Real-time sentiment analysis for customer speech turns.
Uses HuggingFace transformers for local inference.
Maps outputs to debt-collection-relevant labels.
"""

import logging
from typing import Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


# Mapping from HuggingFace emotion labels to our domain labels
EMOTION_MAP = {
    # distilbert-base-uncased-finetuned-sst-2-english
    "POSITIVE": "cooperative",
    "NEGATIVE": "frustration",
    # j-hartmann/emotion-english-distilroberta-base
    "anger": "anger",
    "disgust": "frustration",
    "fear": "frustration",
    "joy": "cooperative",
    "neutral": "neutral",
    "sadness": "frustration",
    "surprise": "neutral",
}

DEBT_ANGER_KEYWORDS = [
    "this is ridiculous", "leave me alone", "stop harassing", "i'll sue",
    "illegal", "lawyer", "report you", "go to hell", "f***", "shut up",
    "stop calling", "never pay", "scam", "fraud", "harassment",
]

COOPERATIVE_KEYWORDS = [
    "i understand", "i can try", "let me check", "i'll see", "sounds fair",
    "okay", "sure", "yes", "i agree", "thank you", "appreciate",
]


class SentimentAnalyzer:
    """
    Analyzes sentiment of customer speech.
    Uses a HuggingFace pipeline with fallback to keyword matching.
    """

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        """Lazy-load the HuggingFace pipeline."""
        try:
            from transformers import pipeline
            logger.info(f"Loading sentiment model: {self.model_name}")
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=1,
                truncation=True,
                max_length=128,
            )
            logger.info("Sentiment model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace model: {e}. Using keyword fallback.")
            self._pipeline = None

    def analyze(self, text: str) -> Dict[str, object]:
        """
        Analyze sentiment of a text segment.

        Returns:
            dict with keys: label (str), score (float), raw (str)
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.5, "raw": "empty"}

        # Keyword override (high-confidence signals)
        keyword_result = self._keyword_override(text)
        if keyword_result:
            return keyword_result

        # Model inference
        if self._pipeline:
            try:
                result = self._pipeline(text[:512])
                # result format: [[{"label": "...", "score": ...}]]
                if result and result[0]:
                    raw_label = result[0][0]["label"]
                    score = result[0][0]["score"]
                    mapped_label = EMOTION_MAP.get(raw_label, "neutral")
                    return {
                        "label": mapped_label,
                        "score": round(score, 4),
                        "raw": raw_label,
                    }
            except Exception as e:
                logger.warning(f"Sentiment model inference failed: {e}")

        # Pure keyword fallback
        return self._keyword_fallback(text)

    def _keyword_override(self, text: str) -> Optional[Dict]:
        """High-confidence keyword detection that overrides model output."""
        text_lower = text.lower()
        for kw in DEBT_ANGER_KEYWORDS:
            if kw in text_lower:
                return {"label": "anger", "score": 0.95, "raw": "keyword_override"}
        return None

    def _keyword_fallback(self, text: str) -> Dict:
        """Simple keyword-based sentiment when model unavailable."""
        text_lower = text.lower()

        anger_score = sum(1 for kw in DEBT_ANGER_KEYWORDS if kw in text_lower)
        coop_score = sum(1 for kw in COOPERATIVE_KEYWORDS if kw in text_lower)

        if anger_score > coop_score:
            return {"label": "anger", "score": min(0.5 + anger_score * 0.1, 0.95), "raw": "keyword"}
        elif coop_score > anger_score:
            return {"label": "cooperative", "score": min(0.5 + coop_score * 0.1, 0.95), "raw": "keyword"}
        else:
            return {"label": "neutral", "score": 0.5, "raw": "keyword"}

    def batch_analyze(self, texts: list) -> list:
        """Analyze a list of text segments."""
        return [self.analyze(t) for t in texts]

    def compute_call_sentiment_summary(self, sentiment_history: list) -> Dict:
        """
        Summarize sentiment across an entire call.

        Args:
            sentiment_history: List of {label, score} dicts from the call
        """
        if not sentiment_history:
            return {"dominant": "neutral", "anger_ratio": 0.0, "cooperative_ratio": 0.0}

        counts = {}
        for entry in sentiment_history:
            label = entry.get("label", "neutral")
            counts[label] = counts.get(label, 0) + 1

        total = len(sentiment_history)
        dominant = max(counts, key=counts.get)

        return {
            "dominant": dominant,
            "anger_ratio": round(counts.get("anger", 0) / total, 3),
            "frustration_ratio": round(counts.get("frustration", 0) / total, 3),
            "cooperative_ratio": round(counts.get("cooperative", 0) / total, 3),
            "neutral_ratio": round(counts.get("neutral", 0) / total, 3),
            "turn_count": total,
        }
