import pandas as pd
import numpy as np
import nltk
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI
import uvicorn
nltk.download('punkt')       # Tokenizer
nltk.download('stopwords')   # Common stopwords
nltk.download('wordnet')     # For lemmatization
nltk.download('omw-1.4')     # Lemmatizer support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
model_path = r"D:\desktop\NeoSoft\SL\ML Sentiment Analysis\sentiment_model"  # folder with your files

app = FastAPI()


@app.get("/")
def home():
    return """Hello!!! Welcome to Sentimental Page...."""

@app.post("/predict")
def predict(sentence):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    nlp_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


    preds = nlp_pipeline(sentence)

    if preds[0]['label'] == "POS":
        return "Positive"
    elif preds[0]['label'] == "NEU":
        return "Neutral"
    else:
        return "Negative"


if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost",port = 8000)










