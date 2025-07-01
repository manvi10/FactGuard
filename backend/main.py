from fastapi import FastAPI, HTTPException, Request
from backend.schemas import NewsItem, NewsOnly, NewsResponse, ExplanationResponse, DualNewsResponse
from backend.utils import predict, predict_fakereal
from backend.explainer import explain, explain_fakereal

import logging

logging.basicConfig(filename="requests.log", level=logging.INFO)

app = FastAPI(
    title="Hybrid RoBERTa Fake News Detector API",
    description="Classifies political news statements using both RoBERTa+metadata and BERT Fake/Real models",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake News Detection API!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return {
        "liar_model": "RoBERTa + metadata (LIAR dataset)",
        "fake_real_model": "BERT base (Kaggle Fake/Real dataset)",
        "version": "v1.0"
    }

# ✅ Combined Predict: Uses both models
@app.post("/predict", response_model=DualNewsResponse)
def classify_both_models(item: NewsItem):
    try:
        text = item.statement + " [SEP] " + item.context

        liar_result = predict(
            statement=item.statement,
            context=item.context,
            party=item.party,
            state=item.state,
            barely_true=item.barely_true,
            false=item.false,
            half_true=item.half_true,
            mostly_true=item.mostly_true,
            pants_on_fire=item.pants_on_fire
        )

        fakereal_result = predict_fakereal(text)

        return {
            "liar_model_prediction": liar_result,
            "fakereal_model_prediction": fakereal_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explain LIAR
@app.post("/explain/liar", response_model=ExplanationResponse)
def explain_news(item: NewsItem):
    try:
        text = item.statement + " [SEP] " + item.context
        explanation = explain(text)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explain BERT-only
@app.post("/explain/fakereal", response_model=ExplanationResponse)
def explain_fakereal_news(item: NewsOnly):
    try:
        return {"explanation": explain_fakereal(item.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logging.info(f"Request: {body.decode('utf-8')}")
    return await call_next(request)
