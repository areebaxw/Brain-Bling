"""
FastAPI Backend for Brain Bling
Provides API endpoints for React frontend to interact with ML models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import os
from typing import List, Optional

app = FastAPI(title="Brain Bling API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory path
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Load models on startup
trained_models = None
ensemble_models = None
distractor_model = None
hint_model = None
onehot_vectorizer = None

def load_models():
    global trained_models, ensemble_models, distractor_model, hint_model, onehot_vectorizer
    try:
        if os.path.exists(f"{MODELS_DIR}/trained_models.pkl"):
            with open(f"{MODELS_DIR}/trained_models.pkl", "rb") as f:
                trained_models = pickle.load(f)
        
        if os.path.exists(f"{MODELS_DIR}/ensemble_models.pkl"):
            with open(f"{MODELS_DIR}/ensemble_models.pkl", "rb") as f:
                ensemble_models = pickle.load(f)
        
        if os.path.exists(f"{MODELS_DIR}/distractor_model.pkl"):
            with open(f"{MODELS_DIR}/distractor_model.pkl", "rb") as f:
                distractor_model = pickle.load(f)
        
        if os.path.exists(f"{MODELS_DIR}/hint_model.pkl"):
            with open(f"{MODELS_DIR}/hint_model.pkl", "rb") as f:
                hint_model = pickle.load(f)
        
        if os.path.exists(f"{MODELS_DIR}/onehot_vectorizer.pkl"):
            with open(f"{MODELS_DIR}/onehot_vectorizer.pkl", "rb") as f:
                onehot_vectorizer = pickle.load(f)
        
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models on startup
load_models()

# Pydantic models
class ArticleInput(BaseModel):
    article: str

class QuestionInput(BaseModel):
    article: str
    question: str
    options: List[str]

class AnswerCheck(BaseModel):
    article: str
    question: str
    selected_option: str
    options: List[str]

@app.get("/")
async def root():
    return {"message": "Brain Bling API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "trained_models": trained_models is not None,
            "ensemble_models": ensemble_models is not None,
            "distractor_model": distractor_model is not None,
            "hint_model": hint_model is not None,
            "onehot_vectorizer": onehot_vectorizer is not None
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    try:
        binary_metrics = pd.read_csv(f"{MODELS_DIR}/binary_classification_metrics.csv")
        ensemble_metrics = pd.read_csv(f"{MODELS_DIR}/ensemble_binary_metrics.csv")
        
        return {
            "binary_metrics": binary_metrics.to_dict(orient="records"),
            "ensemble_metrics": ensemble_metrics.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")

@app.post("/api/generate-question")
async def generate_question(input: ArticleInput):
    """Generate a question from the article using Model A"""
    # Placeholder - implement actual question generation
    return {
        "question": "What is the main idea of the passage?",
        "options": [
            "Option A placeholder",
            "Option B placeholder",
            "Option C placeholder",
            "Option D placeholder"
        ],
        "correct_answer": "B"
    }

@app.post("/api/generate-distractors")
async def generate_distractors(input: QuestionInput):
    """Generate distractors using Model B"""
    # Placeholder - implement actual distractor generation
    return {
        "distractors": [
            "Distractor 1",
            "Distractor 2",
            "Distractor 3"
        ]
    }

@app.post("/api/generate-hints")
async def generate_hints(input: QuestionInput):
    """Generate hints using Model B"""
    # Placeholder - implement actual hint generation
    return {
        "hints": [
            "Hint 1: Look for key information in the passage",
            "Hint 2: Focus on the main topic",
            "Hint 3: Consider the context"
        ]
    }

@app.post("/api/check-answer")
async def check_answer(input: AnswerCheck):
    """Verify the answer using Model A"""
    # Placeholder - implement actual answer verification
    is_correct = input.selected_option == "B"  # Simplified logic
    confidence = 0.75 if is_correct else 0.25
    
    return {
        "is_correct": is_correct,
        "confidence": confidence,
        "correct_answer": "B",
        "explanation": "Model A verification result"
    }

@app.get("/api/sample-article")
async def get_sample_article():
    """Get a sample RACE article"""
    return {
        "article": "Mina joined a reading competition at school. She practiced every evening and learned how to find key ideas in long passages. On the final day, she answered most questions correctly because she focused on the main idea of each paragraph.",
        "question": "What is the main idea of the passage?",
        "options": [
            "Mina wanted to avoid reading practice",
            "Mina improved by practicing reading comprehension",
            "The competition was about drawing pictures",
            "Mina answered questions without reading the passage"
        ],
        "correct_answer": "B"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
