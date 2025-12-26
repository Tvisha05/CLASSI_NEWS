"""
FastAPI Backend for ClassiNews - News Article Classification API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import joblib
import ssl
import string
import json
from datetime import datetime
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):
    """Preprocess text: lowercase, remove punctuation, stem, remove stopwords"""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

LABEL_TO_CATEGORY = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

app = FastAPI(title="ClassiNews API", description="News Article Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
PREPROCESS_PATH = os.path.join(os.path.dirname(__file__), 'preprocess.pkl')

print("Loading model and preprocessor...")
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(PREPROCESS_PATH)
    print("Model and preprocessor loaded successfully!")
    print(f"Model categories: {model.classes_}")
except FileNotFoundError as e:
    print(f"Error: Model files not found. Please train the model first.")
    print(f"Expected files: {MODEL_PATH}, {PREPROCESS_PATH}")
    model = None
    vectorizer = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    vectorizer = None

class NewsArticle(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to ClassiNews API",
        "endpoints": {
            "/predict": "POST - Classify news article",
            "/predict/batch": "POST - Classify multiple articles",
            "/articles/sample": "GET - Get sample articles for dashboard",
            "/health": "GET - Check API health"
        },
        "categories": list(LABEL_TO_CATEGORY.values())
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is None or vectorizer is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "message": "API is running"}


def _predict_with_confidence(text_tfidf):
    """
    Helper to get (mapped_category, confidence_percent) for a single TF-IDF vector.
    Works with models that have either predict_proba or decision_function (e.g. RidgeClassifier).
    """
    raw_label = model.predict(text_tfidf)[0]

    prediction = LABEL_TO_CATEGORY.get(raw_label, str(raw_label))

    confidence = 1.0

    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_tfidf)[0]
            confidence = float(np.max(probabilities))
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(text_tfidf)
            if scores.ndim > 1:
                scores = scores[0]
            exps = np.exp(scores - np.max(scores))
            probs = exps / np.sum(exps)
            confidence = float(np.max(probs))
    except Exception:
        confidence = 1.0

    return prediction, round(confidence * 100, 2)


@app.post("/predict", response_model=PredictionResponse)
def predict_category(article: NewsArticle):
    """
    Predict the category of a news article
    
    Args:
        article: NewsArticle object containing the text
    
    Returns:
        PredictionResponse with category and confidence
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    if not article.text or not article.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    
    try:
        processed_text = preprocess(article.text)

        text_tfidf = vectorizer.transform([processed_text])

        prediction, confidence = _predict_with_confidence(text_tfidf)

        return PredictionResponse(
            category=prediction,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(articles: List[NewsArticle]):
    """
    Predict categories for multiple articles (for dashboard)
    
    Args:
        articles: List of NewsArticle objects
    
    Returns:
        List of predictions with text preview, category, and confidence
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    results = []
    for article in articles:
        if not article.text or not article.text.strip():
            continue

        try:
            processed_text = preprocess(article.text)
            text_tfidf = vectorizer.transform([processed_text])
            prediction, confidence = _predict_with_confidence(text_tfidf)

            results.append({
                "text": article.text[:200] + "..." if len(article.text) > 200 else article.text,
                "full_text": article.text,
                "category": prediction,
                "confidence": confidence
            })
        except Exception:
            continue

    return {"results": results, "count": len(results)}

SAVED_ARTICLES_FILE = os.path.join(os.path.dirname(__file__), 'saved_articles.json')

class SavedArticle(BaseModel):
    id: str
    text: str
    category: str
    confidence: float
    timestamp: str
    isSaved: bool = True

def load_saved_articles_from_file():
    if os.path.exists(SAVED_ARTICLES_FILE):
        try:
            with open(SAVED_ARTICLES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading saved articles: {e}")
            return []
    return []

def save_articles_to_file(articles_list):
    try:
        with open(SAVED_ARTICLES_FILE, 'w') as f:
            json.dump(articles_list, f, indent=2)
    except Exception as e:
        print(f"Error saving articles: {e}")

@app.get("/articles/saved")
def get_saved_articles():
    """Get all saved articles from the server"""
    return load_saved_articles_from_file()

@app.post("/articles/save")
def save_article(article: SavedArticle):
    """Save an article to the server"""
    articles = load_saved_articles_from_file()
    if not any(a['id'] == article.id for a in articles):
        articles.insert(0, article.dict())
        save_articles_to_file(articles)
    return {"status": "success", "message": "Article saved successfully"}

@app.delete("/articles/saved/{article_id}")
def delete_saved_article(article_id: str):
    """Delete a saved article"""
    articles = load_saved_articles_from_file()
    articles = [a for a in articles if a['id'] != article_id]
    save_articles_to_file(articles)
    return {"status": "success", "message": "Article deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
