# 🛡️ SpamShield

A machine learning-powered spam email detection system built with FastAPI, scikit-learn, and Tailwind CSS.

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **ML**: scikit-learn (TF-IDF + Logistic Regression)
- **Dataset**: SMS Spam Collection (~5,500 samples)
- **Frontend**: HTML + Tailwind CSS + Vanilla JS
- **Container**: Docker + Docker Compose

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.65% |
| Precision (Spam) | 100% |
| Recall (Spam) | 90% |
| F1 (Spam) | 95% |

## Running Locally

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Install dependencies
uv sync

# 3. Train the model
uv run python app/model/trainer.py

# 4. Start the server
uv run uvicorn main:app --host 0.0.0.0 --port 7000 --reload
```

Open [http://localhost:7000](http://localhost:7000)

## Running with Docker

```bash
docker compose up --build
```

Open [http://localhost:7000](http://localhost:7000)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Predict if text is spam |
| `GET`  | `/api/health` | Health check |
| `GET`  | `/docs` | Interactive API docs (Swagger) |

### Example Request

```bash
curl -X POST http://localhost:7000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free iPhone. Click here to claim."}'
```

### Example Response

```json
{
  "is_spam": true,
  "label": "SPAM",
  "confidence": 99.87,
  "spam_probability": 99.87,
  "ham_probability": 0.13
}
```
