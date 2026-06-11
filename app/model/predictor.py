from .trainer import load

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load()
    return _pipeline


def predict(text: str) -> dict:
    pipeline = get_pipeline()
    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]

    confidence = float(max(proba)) * 100
    is_spam = bool(label == 1)

    return {
        "is_spam": is_spam,
        "label": "SPAM" if is_spam else "HAM",
        "confidence": round(confidence, 2),
        "spam_probability": round(float(proba[1]) * 100, 2),
        "ham_probability": round(float(proba[0]) * 100, 2),
    }
