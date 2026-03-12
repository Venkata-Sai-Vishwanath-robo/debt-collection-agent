"""
risk_model.py
-------------
Scikit-learn based repayment risk prediction model.
Trained on synthetic data; swap with real data in production.

Predicts: probability that a customer will repay their debt.
"""

import os
import json
import logging
import pickle
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

MODEL_PATH = "ml/risk_model.pkl"
SCALER_PATH = "ml/risk_scaler.pkl"


def _generate_synthetic_training_data(n_samples: int = 1000):
    """
    Generate synthetic training data for the risk model.
    In production, replace with real labeled call data.
    
    Features:
        0: dominant_sentiment_encoded (0=anger, 1=frustration, 2=neutral, 3=cooperative)
        1: anger_ratio (0.0-1.0)
        2: refusal_count (0-5+)
        3: payment_intent (0 or 1)
        4: dispute_raised (0 or 1)
        5: already_paid_claimed (0 or 1)
        6: call_duration_seconds (normalized)
        7: turn_count
        8: cooperative_ratio (0.0-1.0)
        9: opt_out_requested (0 or 1)

    Label: repaid (1) or not (0)
    """
    rng = np.random.RandomState(42)
    X = []
    y = []

    for _ in range(n_samples):
        # Cooperative customer — likely to pay
        if rng.random() < 0.4:
            sentiment = rng.choice([2, 3], p=[0.3, 0.7])
            anger_ratio = rng.uniform(0, 0.2)
            refusals = rng.randint(0, 2)
            payment_intent = rng.choice([0, 1], p=[0.1, 0.9])
            dispute = 0
            already_paid = rng.choice([0, 1], p=[0.8, 0.2])
            duration = rng.uniform(60, 360)
            turns = rng.randint(4, 20)
            coop_ratio = rng.uniform(0.5, 1.0)
            opt_out = 0
            label = 1 if rng.random() < 0.82 else 0

        # Resistant customer — unlikely to pay
        elif rng.random() < 0.35:
            sentiment = rng.choice([0, 1], p=[0.5, 0.5])
            anger_ratio = rng.uniform(0.3, 0.9)
            refusals = rng.randint(2, 6)
            payment_intent = rng.choice([0, 1], p=[0.85, 0.15])
            dispute = rng.choice([0, 1], p=[0.5, 0.5])
            already_paid = rng.choice([0, 1], p=[0.6, 0.4])
            duration = rng.uniform(30, 180)
            turns = rng.randint(2, 12)
            coop_ratio = rng.uniform(0.0, 0.3)
            opt_out = rng.choice([0, 1], p=[0.5, 0.5])
            label = 1 if rng.random() < 0.12 else 0

        # Neutral / mixed customer
        else:
            sentiment = rng.choice([1, 2], p=[0.4, 0.6])
            anger_ratio = rng.uniform(0.1, 0.5)
            refusals = rng.randint(0, 4)
            payment_intent = rng.choice([0, 1], p=[0.5, 0.5])
            dispute = rng.choice([0, 1], p=[0.7, 0.3])
            already_paid = rng.choice([0, 1], p=[0.75, 0.25])
            duration = rng.uniform(45, 300)
            turns = rng.randint(3, 15)
            coop_ratio = rng.uniform(0.2, 0.6)
            opt_out = 0
            label = 1 if rng.random() < 0.45 else 0

        X.append([
            sentiment,
            anger_ratio,
            refusals,
            int(payment_intent),
            int(dispute),
            int(already_paid),
            duration / 600.0,  # normalize to ~0-1
            turns / 30.0,
            coop_ratio,
            int(opt_out),
        ])
        y.append(label)

    return np.array(X), np.array(y)


def train_and_save_model():
    """Train the risk model on synthetic data and persist it."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    logger.info("Training risk prediction model...")
    X, y = _generate_synthetic_training_data(n_samples=2000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # Cross-val score
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
    logger.info(f"Risk model CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    os.makedirs("ml", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"Model saved to {MODEL_PATH}")
    return model, scaler


class RiskPredictor:
    """
    Predicts repayment probability after a debt collection call.

    Inputs are derived from ConversationManager state and transcript.
    """

    SENTIMENT_ENCODING = {
        "anger": 0,
        "frustration": 1,
        "neutral": 2,
        "cooperative": 3,
    }

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_or_train()

    def _load_or_train(self):
        """Load persisted model or train a new one."""
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info("Risk model loaded from disk.")
                return
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Retraining.")

        self.model, self.scaler = train_and_save_model()

    def _extract_features(self, state: Dict[str, Any], transcript: List[Dict]) -> np.ndarray:
        """Convert conversation state into a feature vector."""
        dominant_sentiment = state.get("dominant_sentiment", "neutral")
        sentiment_enc = self.SENTIMENT_ENCODING.get(dominant_sentiment, 2)

        # Compute anger_ratio from sentiment history
        history = state.get("sentiment_history", [])
        if history:
            anger_count = sum(1 for h in history if h.get("label") in ("anger", "frustration"))
            anger_ratio = anger_count / len(history)
            coop_count = sum(1 for h in history if h.get("label") == "cooperative")
            coop_ratio = coop_count / len(history)
        else:
            anger_ratio = 0.0
            coop_ratio = 0.5

        features = [
            sentiment_enc,
            anger_ratio,
            min(state.get("refusal_count", 0), 6),
            int(state.get("payment_intent", False)),
            int(state.get("dispute_raised", False)),
            int(state.get("already_paid_claimed", False)),
            min(state.get("call_duration_seconds", 0), 600) / 600.0,
            min(state.get("turn_count", 0), 30) / 30.0,
            coop_ratio,
            int(state.get("opt_out_requested", False)),
        ]

        return np.array(features).reshape(1, -1)

    def predict(self, state: Dict[str, Any], transcript: List[Dict]) -> float:
        """
        Predict repayment probability.

        Returns:
            float between 0.0 (very unlikely to repay) and 1.0 (very likely)
        """
        try:
            features = self._extract_features(state, transcript)
            features_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(features_scaled)[0][1]
            logger.info(f"Repayment risk score: {proba:.4f}")
            return float(proba)
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            # Fallback: heuristic score
            return self._heuristic_score(state)

    def _heuristic_score(self, state: Dict[str, Any]) -> float:
        """Simple heuristic fallback when model fails."""
        score = 0.5
        if state.get("payment_intent"):
            score += 0.25
        if state.get("dispute_raised"):
            score -= 0.2
        if state.get("already_paid_claimed"):
            score -= 0.1
        score -= state.get("refusal_count", 0) * 0.08
        if state.get("dominant_sentiment") == "cooperative":
            score += 0.1
        elif state.get("dominant_sentiment") == "anger":
            score -= 0.15
        return round(max(0.0, min(1.0, score)), 4)

    def explain(self, state: Dict[str, Any], transcript: List[Dict]) -> Dict:
        """Return feature values alongside the prediction for explainability."""
        features = self._extract_features(state, transcript)
        score = self.predict(state, transcript)

        feature_names = [
            "sentiment_encoded", "anger_ratio", "refusal_count",
            "payment_intent", "dispute_raised", "already_paid_claimed",
            "call_duration_norm", "turn_count_norm", "cooperative_ratio",
            "opt_out_requested",
        ]

        return {
            "repayment_probability": round(score, 4),
            "risk_label": "LOW_RISK" if score >= 0.75 else ("MEDIUM_RISK" if score >= 0.45 else "HIGH_RISK"),
            "features": {name: round(float(val), 4) for name, val in zip(feature_names, features[0])},
        }
