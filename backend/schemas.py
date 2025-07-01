from pydantic import BaseModel
from typing import List, Tuple

class NewsItem(BaseModel):
    statement: str
    context: str
    party: str
    state: str
    barely_true: int
    false: int
    half_true: int
    mostly_true: int
    pants_on_fire: int

class NewsOnly(BaseModel):
    text: str

class NewsResponse(BaseModel):
    prediction: str
    confidence: float

class ExplanationResponse(BaseModel):
    explanation: List[Tuple[str, float]]

class DualNewsResponse(BaseModel):
    liar_model_prediction: NewsResponse
    fakereal_model_prediction: NewsResponse
