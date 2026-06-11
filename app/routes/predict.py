from fastapi import APIRouter
from pydantic import BaseModel
from app.model.predictor import predict

router = APIRouter()


class EmailRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    is_spam: bool
    label: str
    confidence: float
    spam_probability: float
    ham_probability: float


@router.post("/predict", response_model=PredictionResponse)
def predict_email(request: EmailRequest):
    if not request.text.strip():
        return PredictionResponse(
            is_spam=False,
            label="HAM",
            confidence=0.0,
            spam_probability=0.0,
            ham_probability=0.0,
        )
    result = predict(request.text)
    return PredictionResponse(**result)


@router.head("/health")
@router.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}
