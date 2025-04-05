from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import os
import json
from recommender import SHLRecommender
from pydantic import BaseModel

# Initialize the app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends SHL assessments based on natural language job descriptions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender
data_path = "shl_assessments.json" if os.path.exists("shl_assessments.json") else "sample_assessments.json"
if not os.path.exists(data_path):
    from recommender import create_sample_data
    create_sample_data()
    data_path = "sample_assessments.json"

recommender = SHLRecommender(data_path=data_path)

# Define request model
class RecommendationRequest(BaseModel):
    query: str
    max_duration: Optional[int] = None
    top_k: int = 5

@app.get("/")
async def root():
    """Root endpoint - provides a simple welcome message"""
    return {
        "message": "Welcome to the SHL Assessment Recommendation API",
        "documentation": "/docs",
        "endpoints": [
            {
                "path": "/recommend",
                "description": "Get assessment recommendations",
                "methods": ["GET", "POST"]
            }
        ]
    }

@app.get("/recommend")
async def recommend(
    query: str = Query(..., description="Natural language query about job requirements"),
    max_duration: Optional[int] = Query(None, description="Maximum duration in minutes"),
    top_k: int = Query(5, description="Number of recommendations to return")
):
    """
    Get assessment recommendations based on a job description query
    
    Args:
        query: Natural language query describing the job role or skills needed
        max_duration: Maximum assessment duration in minutes (optional)
        top_k: Number of recommendations to return (default: 5)
    
    Returns:
        List of recommended assessments with relevance scores
    """
    try:
        results = recommender.recommend(query=query, max_duration=max_duration, top_k=top_k)
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/recommend")
async def recommend_post(request: RecommendationRequest):
    """
    Get assessment recommendations based on a job description query (POST method)
    
    Request body:
        query: Natural language query describing the job role or skills needed
        max_duration: Maximum assessment duration in minutes (optional)
        top_k: Number of recommendations to return (default: 5)
    
    Returns:
        List of recommended assessments with relevance scores
    """
    try:
        results = recommender.recommend(
            query=request.query,
            max_duration=request.max_duration,
            top_k=request.top_k
        )
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the underlying model and data"""
    return {
        "model_type": "TF-IDF + Keyword Matching + Gemini Reranking",
        "embeddings_model": "scikit-learn TfidfVectorizer",
        "assessments_count": len(recommender.assessments),
        "has_gemini_integration": bool(os.getenv("GOOGLE_API_KEY")),
        "data_source": data_path
    }

@app.get("/assessments")
async def get_assessments():
    """Get all available assessments"""
    try:
        assessments = [
            {
                "name": a.name,
                "url": a.url,
                "test_type": a.test_type,
                "remote_testing": a.remote_testing,
                "adaptive_irt": a.adaptive_irt,
                "duration": a.duration
            }
            for a in recommender.assessments
        ]
        return {"assessments": assessments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assessments: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 