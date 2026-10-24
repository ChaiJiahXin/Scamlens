import os
import re
import joblib

# Lightweight ML model: TF-IDF + Logistic Regression (no torch needed).
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, "model.pkl"))

_vec = model.steps[0][1]
_clf = model.steps[-1][1]
_vocab = _vec.vocabulary_
_coef = _clf.coef_[0]


def predict_scam(text):
    """Return scam probability and the top scam-related terms found in the text."""
    if not text or not text.strip():
        return None

    prob = float(model.predict_proba([text])[0][1])

    # Terms from the message that push the model toward "scam" (highest weight first).
    words = set(re.findall(r"[a-z0-9']+", text.lower()))
    scored = []
    for w in words:
        i = _vocab.get(w)
        if i is not None and _coef[i] > 0:
            scored.append((_coef[i], w))
    scored.sort(reverse=True)
    top_terms = [w for _, w in scored[:3]]

    return {"probability": prob, "top_terms": top_terms}


def to_finding(ml_result):
    """Convert a model result into the finding format used by the main app."""
    if not ml_result or ml_result["probability"] < 0.5:
        return None

    return {
        "key": "ml_cue",
        "type": "🧠 ML Model Detection",
        "evidence": ", ".join(ml_result["top_terms"]) or "Pattern matching",
        "all_matches": ml_result["top_terms"],
        "explanation": f"The local ML model found this text has a {ml_result['probability']:.0%} probability of being a scam based on historical patterns.",
        "severity": "Medium",
    }
