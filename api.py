import os
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from recommender import get_recommender, RecommendationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API models
class QueryInput(BaseModel):
    """Query input for recommendation API"""
    query: str = Field(..., description="Query text or job description")
    num_recommendations: int = Field(3, description="Number of recommendations to return")

class RecommendationResponse(BaseModel):
    """Response model for recommendation API"""
    assessment_name: str = Field(..., description="Name of the SHL assessment")
    assessment_url: str = Field(..., description="URL to the assessment page")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    explanation: Optional[str] = Field(None, description="Explanation of why this assessment is recommended")
    duration: Optional[int] = Field(None, description="Duration of the assessment in minutes")
    remote_testing: bool = Field(False, description="Whether remote testing is supported")
    adaptive_irt: bool = Field(False, description="Whether adaptive/IRT testing is supported")
    test_type: List[str] = Field([], description="Types of tests included in the assessment")

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions or queries",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Get recommendations with GET (query params) - used by the Streamlit app
@app.get("/recommend", 
         response_model=List[RecommendationResponse],
         responses={
             200: {"description": "Successful response"},
             422: {"model": ErrorResponse, "description": "Validation error"},
             500: {"model": ErrorResponse, "description": "Server error"}
         },
         tags=["Recommendations"])
async def get_recommendations_simple(
    query: str = Query(..., description="Query text or job description"),
    top_k: int = Query(10, description="Number of recommendations to return", ge=1, le=20)
):
    """
    Get SHL assessment recommendations based on a query or job description.
    Used by the Streamlit app.
    
    - **query**: The query text or job description to match against assessments
    - **top_k**: Number of recommendations to return (default: 10)
    
    Returns a list of recommended assessments with relevance scores and metadata.
    """
    try:
        # Get recommender instance
        recommender = get_recommender()
        
        # Get recommendations
        recommendations = await recommender.get_recommendations(query, top_k)
        
        # Convert to response model
        response = [
            RecommendationResponse(
                assessment_name=rec.assessment_name,
                assessment_url=rec.assessment_url,
                relevance_score=rec.relevance_score,
                explanation=rec.explanation,
                duration=rec.duration,
                remote_testing=rec.remote_testing,
                adaptive_irt=rec.adaptive_irt,
                test_type=rec.test_type
            )
            for rec in recommendations
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Get recommendations with GET (query params)
@app.get("/recommendations", 
         response_model=List[RecommendationResponse],
         responses={
             200: {"description": "Successful response"},
             422: {"model": ErrorResponse, "description": "Validation error"},
             500: {"model": ErrorResponse, "description": "Server error"}
         },
         tags=["Recommendations"])
async def get_recommendations(
    query: str = Query(..., description="Query text or job description"),
    num_recommendations: int = Query(3, description="Number of recommendations to return", ge=1, le=10)
):
    """
    Get SHL assessment recommendations based on a query or job description.
    
    - **query**: The query text or job description to match against assessments
    - **num_recommendations**: Number of recommendations to return (default: 3)
    
    Returns a list of recommended assessments with relevance scores and metadata.
    """
    try:
        # Get recommender instance
        recommender = get_recommender()
        
        # Get recommendations
        recommendations = await recommender.get_recommendations(query, num_recommendations)
        
        # Check if we got recommendations
        if not recommendations:
            # Try to generate a generic response
            generic_response = await recommender.generate_generic_response(query)
            
            # If we got a generic response, return it as an error
            if generic_response:
                raise HTTPException(status_code=404, detail={
                    "message": "No matching assessments found",
                    "response": generic_response
                })
            else:
                raise HTTPException(status_code=404, detail="No matching assessments found")
        
        # Convert to response model
        response = [
            RecommendationResponse(
                assessment_name=rec.assessment_name,
                assessment_url=rec.assessment_url,
                relevance_score=rec.relevance_score,
                explanation=rec.explanation,
                duration=rec.duration,
                remote_testing=rec.remote_testing,
                adaptive_irt=rec.adaptive_irt,
                test_type=rec.test_type
            )
            for rec in recommendations
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Get recommendations with POST (JSON body)
@app.post("/recommendations", 
          response_model=List[RecommendationResponse],
          responses={
              200: {"description": "Successful response"},
              422: {"model": ErrorResponse, "description": "Validation error"},
              500: {"model": ErrorResponse, "description": "Server error"}
          },
          tags=["Recommendations"])
async def post_recommendations(input_data: QueryInput):
    """
    Get SHL assessment recommendations based on a query or job description (POST method).
    
    - **input_data**: JSON object with query and optional parameters
    
    Returns a list of recommended assessments with relevance scores and metadata.
    """
    try:
        # Get recommender instance
        recommender = get_recommender()
        
        # Get recommendations
        recommendations = await recommender.get_recommendations(
            input_data.query, 
            input_data.num_recommendations
        )
        
        # Check if we got recommendations
        if not recommendations:
            # Try to generate a generic response
            generic_response = await recommender.generate_generic_response(input_data.query)
            
            # If we got a generic response, return it as an error
            if generic_response:
                raise HTTPException(status_code=404, detail={
                    "message": "No matching assessments found",
                    "response": generic_response
                })
            else:
                raise HTTPException(status_code=404, detail="No matching assessments found")
        
        # Convert to response model
        response = [
            RecommendationResponse(
                assessment_name=rec.assessment_name,
                assessment_url=rec.assessment_url,
                relevance_score=rec.relevance_score,
                explanation=rec.explanation,
                duration=rec.duration,
                remote_testing=rec.remote_testing,
                adaptive_irt=rec.adaptive_irt,
                test_type=rec.test_type
            )
            for rec in recommendations
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Get all assessments
@app.get("/assessments", 
         response_model=List[Dict[str, Any]],
         responses={
             200: {"description": "Successful response"},
             500: {"model": ErrorResponse, "description": "Server error"}
         },
         tags=["Assessments"])
async def get_all_assessments():
    """
    Get all available SHL assessments.
    
    Returns a list of all assessments in the system.
    """
    try:
        # Get recommender instance
        recommender = get_recommender()
        
        # Return all assessments
        return recommender.assessments
        
    except Exception as e:
        logger.error(f"Error getting assessments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# On startup event - load assessments and initialize recommender
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        # Check if assessments file exists
        if not os.path.exists("shl_assessments.json"):
            logger.warning("No assessments file found, will create sample data")
            # Attempt to import scraper
            try:
                from scraper import scrape_all_shl_assessments
                logger.info("Running scraper to get assessment data")
                assessments, _ = await scrape_all_shl_assessments()
                logger.info(f"Scraped {len(assessments)} assessments")
            except ImportError:
                logger.error("Scraper module not found, creating empty assessments file")
                with open("shl_assessments.json", "w") as f:
                    json.dump([], f)
        
        # Initialize recommender
        recommender = get_recommender()
        logger.info(f"Recommender initialized with {len(recommender.assessments)} assessments")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

def run_api():
    """Run the API server"""
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    run_api() 