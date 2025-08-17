import pandas as pd
import numpy as np
import nltk
import warnings
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

warnings.filterwarnings('ignore')

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Model path
model_path = r"D:\desktop\NeoSoft\SL\ML Sentiment Analysis\sentiment_model"

app = FastAPI()

# Enable CORS (important if you call API from JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (your HTML/JS/CSS)
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Root route → serve index.html
@app.get("/")
def read_root():
    return FileResponse("templates/index.html")


# Input schema for prediction
class InputText(BaseModel):
    text: str


@app.post("/predict")
def predict(data: InputText):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    preds = nlp_pipeline(data.text)

    if preds[0]['label'] == "POS":
        return {"sentiment": "Positive"}
    elif preds[0]['label'] == "NEU":
        return {"sentiment": "Neutral"}
    else:
        return {"sentiment": "Negative"}


if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000)
