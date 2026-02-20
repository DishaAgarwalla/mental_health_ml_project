from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import sys
import numpy as np
from typing import Optional, List

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_text
from src.database import init_db, log_prediction, get_recent_predictions, get_statistics

app = FastAPI(title="Mental Health Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
model_path = os.path.join(BASE_DIR, "model", "model.pkl")

# Load model
try:
    model = pickle.load(open(model_path, "rb"))
    print("✅ Model loaded successfully")
    print(f"✅ Model type: {type(model)}")
    if hasattr(model, 'named_steps'):
        print(f"✅ Pipeline steps: {list(model.named_steps.keys())}")
except FileNotFoundError as e:
    print(f"❌ Error loading model: {e}")
    print("Please train the model first using: python src/train_model.py")
    model = None
except Exception as e:
    print(f"❌ Unexpected error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Request/Response models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int  # 0 = normal, 1 = anxiety
    confidence: float
    label: str
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_predictions: Optional[int] = None
    model_type: Optional[str] = None

class HistoryItem(BaseModel):
    id: int
    text: str
    prediction: str
    confidence: float
    timestamp: str

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_db()
    print("✅ Database initialized")

@app.get("/", response_model=HealthResponse)
def home():
    """Health check endpoint"""
    from src.database import get_prediction_count
    
    model_type = None
    if model is not None:
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            model_type = type(model.named_steps['clf']).__name__
        else:
            model_type = type(model).__name__
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        total_predictions=get_prediction_count(),
        model_type=model_type
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict if text indicates anxiety
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")
    
    try:
        # Clean the text
        cleaned_text = clean_text(request.text)
        print(f"Original: '{request.text}'")
        print(f"Cleaned: '{cleaned_text}'")
        
        # Check if cleaned text is empty
        if not cleaned_text or cleaned_text.strip() == "":
            # Return default response for empty text
            return PredictionResponse(
                prediction=0,
                confidence=0.5,
                label="normal",
                message="✅ Empty or invalid text"
            )
        
        # For scikit-learn pipeline, pass as a list of strings
        text_list = [cleaned_text]
        
        # Get prediction
        pred = model.predict(text_list)[0]
        prediction = int(pred)
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_list)[0]
            confidence = float(proba[1]) if prediction == 1 else float(proba[0])
        else:
            confidence = 0.5
        
        print(f"Prediction: {prediction}, Confidence: {confidence}")
        
        # Create response
        if prediction == 1:
            label = "anxiety"
            message = "⚠ Anxiety detected in the text"
        else:
            label = "normal"
            message = "✅ Normal statement"
        
        # Log to database
        try:
            log_prediction(
                text=request.text[:500],
                prediction=label,
                confidence=confidence
            )
        except Exception as db_error:
            print(f"Database error: {db_error}")
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            label=label,
            message=message
        )
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/history", response_model=List[HistoryItem])
def get_history(limit: int = 10):
    """Get recent prediction history"""
    return get_recent_predictions(limit)

@app.get("/stats")
def get_stats():
    """Get prediction statistics"""
    return get_statistics()

@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": type(model.named_steps['clf']).__name__ if hasattr(model, 'named_steps') else type(model).__name__,
        "vectorizer": "TfidfVectorizer",
        "total_features": 0,
        "pipeline_steps": list(model.named_steps.keys()) if hasattr(model, 'named_steps') else []
    }
    
    # Get feature names if available
    try:
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['tfidf'], 'get_feature_names_out'):
            features = model.named_steps['tfidf'].get_feature_names_out()
            info["total_features"] = len(features)
            info["sample_features"] = features[:10].tolist()
    except Exception as e:
        print(f"Error getting features: {e}")
    
    return info

@app.get("/debug-predict")
def debug_predict(text: str = "I am happy today"):
    """Debug endpoint that returns more information"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Clean the text
        cleaned = clean_text(text)
        
        result = {
            "original_text": text,
            "cleaned_text": cleaned,
            "model_type": str(type(model)),
            "predictions": {}
        }
        
        # Try as list
        try:
            text_list = [cleaned]
            pred_list = int(model.predict(text_list)[0])
            proba_list = model.predict_proba(text_list)[0].tolist()
            result["predictions"]["as_list"] = {
                "prediction": pred_list,
                "probabilities": proba_list,
                "label": "anxiety" if pred_list == 1 else "normal"
            }
        except Exception as e:
            result["predictions"]["as_list_error"] = str(e)
        
        # Get model info
        if hasattr(model, 'named_steps'):
            result["model_info"] = {
                "steps": list(model.named_steps.keys()),
            }
            if hasattr(model.named_steps['tfidf'], 'get_feature_names_out'):
                try:
                    result["model_info"]["sample_features"] = model.named_steps['tfidf'].get_feature_names_out()[:5].tolist()
                except:
                    pass
        
        return result
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}