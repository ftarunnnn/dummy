from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class TabularPredictRequest(BaseModel):
    Age: int = Field(..., example=45, ge=18, le=100)
    Systolic_BP: int = Field(..., example=135, ge=80, le=220)
    Cholesterol: int = Field(..., example=215, ge=100, le=400)
    Max_Heart_Rate: int = Field(..., example=160, ge=60, le=220)
    Glucose_Level: int = Field(..., example=105, ge=60, le=300)
    BMI: float = Field(..., example=27.4, ge=12.0, le=55.0)
    Exercise_Angina: int = Field(..., example=0, ge=0, le=1)
    ST_Depression: float = Field(..., example=0.8, ge=0.0, le=6.0)

class TabularPredictResponse(BaseModel):
    risk_level: str
    risk_score_pct: float
    probabilities: Dict[str, float]
    feature_importances: Dict[str, float]

class NLPPredictRequest(BaseModel):
    text: str = Field(..., example="The AI diagnostic model gave fast and accurate predictions.")

class NLPPredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class VisionPredictResponse(BaseModel):
    class_name: str
    confidence: float
    class_probabilities: Dict[str, float]
    diagnostic_recommendation: str

class MetricsResponse(BaseModel):
    status: str
    tabular_accuracy: float
    tabular_f1: float
    vision_accuracy: float
    nlp_accuracy: float
