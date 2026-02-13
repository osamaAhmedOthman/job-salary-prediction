from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Salary Prediction API",
    description="Predict job salaries based on job characteristics",
    version="1.0.0"
)

# Load the trained model
MODEL_PATH = r"/app/model/salary_predictor.pkl"
model = joblib.load(MODEL_PATH)

# Define input data structure
class SalaryInput(BaseModel):
    work_year: int
    experience_encoded: int
    is_full_time: int
    is_remote: int
    is_hybrid: int
    company_size_encoded: int
    job_title_Data_Analyst: int
    job_title_Data_Engineer: int
    job_title_Data_Scientist: int
    job_title_Engineer: int
    job_title_Machine_Learning_Engineer: int
    job_title_Manager: int
    job_title_Other: int
    job_title_Software_Engineer: int
    company_location_AU: int
    company_location_CA: int
    company_location_GB: int
    company_location_Other: int
    company_location_US: int

    class Config:
        json_schema_extra = {
            "example": {
                "work_year": 2024,
                "experience_encoded": 2,
                "is_full_time": 1,
                "is_remote": 0,
                "is_hybrid": 0,
                "company_size_encoded": 1,
                "job_title_Data_Analyst": 1,
                "job_title_Data_Engineer": 0,
                "job_title_Data_Scientist": 0,
                "job_title_Engineer": 0,
                "job_title_Machine_Learning_Engineer": 0,
                "job_title_Manager": 0,
                "job_title_Other": 0,
                "job_title_Software_Engineer": 0,
                "company_location_AU": 0,
                "company_location_CA": 0,
                "company_location_GB": 0,
                "company_location_Other": 0,
                "company_location_US": 1
            }
        }

# Define output structure
class SalaryOutput(BaseModel):
    predicted_salary_usd: float
    prediction_confidence: str

# Root endpoint
@app.get("/")
def read_root():
    """Welcome message"""
    return {
        "message": "Welcome to Salary Prediction API",
        "docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if API is running"""
    return {"status": "API is running successfully"}

# Prediction endpoint
@app.post("/predict", response_model=SalaryOutput)
def predict_salary(data: SalaryInput):
    """
    Predict salary based on job characteristics.
    
    The model expects input as a DataFrame with column names matching training data.
    """
    
    try:
        # Create features as a dictionary with EXACT column names from training
        features_dict = {
            'work_year': data.work_year,
            'experience_encoded': data.experience_encoded,
            'is_full_time': data.is_full_time,
            'is_remote': data.is_remote,
            'is_hybrid': data.is_hybrid,
            'company_size_encoded': data.company_size_encoded,
            'job_title_Data Analyst': data.job_title_Data_Analyst,  # Note: SPACE not underscore
            'job_title_Data Engineer': data.job_title_Data_Engineer,  # Note: SPACE
            'job_title_Data Scientist': data.job_title_Data_Scientist,  # Note: SPACE
            'job_title_Engineer': data.job_title_Engineer,
            'job_title_Machine Learning Engineer': data.job_title_Machine_Learning_Engineer,  # SPACES
            'job_title_Manager': data.job_title_Manager,
            'job_title_Other': data.job_title_Other,
            'job_title_Software Engineer': data.job_title_Software_Engineer,  # Note: SPACE
            'company_location_AU': data.company_location_AU,
            'company_location_CA': data.company_location_CA,
            'company_location_GB': data.company_location_GB,
            'company_location_Other': data.company_location_Other,
            'company_location_US': data.company_location_US
        }
        
        # Convert to DataFrame (ColumnTransformer requires DataFrame)
        X_input = pd.DataFrame([features_dict])
        
        # Make prediction (returns log salary)
        log_prediction = model.predict(X_input)[0]
        
        # Convert from log scale to actual salary USD
        predicted_salary = np.expm1(log_prediction)
        
        # Determine confidence based on salary range
        if predicted_salary < 50000:
            confidence = "Low"
        elif predicted_salary > 200000:
            confidence = "Medium"
        else:
            confidence = "High"
        
        return {
            "predicted_salary_usd": round(predicted_salary, 2),
            "prediction_confidence": confidence
        }
    
    except Exception as e:
        return {
            "predicted_salary_usd": 0,
            "prediction_confidence": f"Error: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)