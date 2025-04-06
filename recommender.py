import json
import numpy as np
import pandas as pd
# Removed: from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import random
import asyncio
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import logging
import re
import time

import google.generativeai as genai
# Removed: from langchain_chroma import Chroma
# Removed: from chromadb.config import Settings
# Removed: from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables. Gemini API will not be available.")

# Define recommendation result schema
class RecommendationResult(BaseModel):
    assessment_name: str
    assessment_url: str
    relevance_score: float
    explanation: Optional[str] = None
    duration: Optional[int] = None
    remote_testing: bool = False
    adaptive_irt: bool = False
    test_type: List[str] = Field(default_factory=list)

class SHLRecommender:
    """
    SHL Assessment Recommender using Google Gemini API and text similarity
    
    This class uses Google's Gemini API to recommend SHL assessments based on
    job descriptions or user queries. It uses a combination of text similarity
    matching and LLM-powered recommendations.
    """
    
    # Prompt template for Gemini AI
    SYSTEM_PROMPT = """You are an expert in SHL assessments and pre-employment testing.
Your task is to recommend the most appropriate SHL assessments for a specific job role or query.
Based on the job description or query, analyze what skills need to be assessed and recommend the most relevant assessments.
Consider the following factors:
- Cognitive abilities required (numerical, verbal, logical, etc.)
- Personality traits that would be important
- Technical knowledge needed
- Required soft skills and competencies
- Leadership requirements if applicable

You will be provided with a set of available SHL assessments in JSON format.
Use ONLY assessments from this list when making your recommendations.
"""

    PROMPT_TEMPLATE = """
Analyze the following job description or query:
---
{query}
---

Based on this information, recommend the top 3 most appropriate SHL assessments from the provided list.
The available SHL assessments are:
{assessment_info}

For each assessment you recommend, provide:
1. The exact name of the assessment as it appears in the list
2. A score from 0.0 to 1.0 indicating relevance to the job or query (higher is more relevant)
3. A 1-2 sentence explanation of why this assessment is appropriate

Format your response as a JSON array like this:
```json
[
  {
    "assessment_name": "Name of first assessment",
    "relevance_score": 0.95,
    "explanation": "Explanation for first assessment"
  },
  {
    "assessment_name": "Name of second assessment",
    "relevance_score": 0.85,
    "explanation": "Explanation for second assessment"
  },
  {
    "assessment_name": "Name of third assessment",
    "relevance_score": 0.75,
    "explanation": "Explanation for third assessment"
  }
]
```

Only include assessments from the provided list. Return EXACTLY 3 recommendations.
"""

    FALLBACK_PROMPT = """
You are an expert in SHL assessments. Provide a concise response to this query about pre-employment testing:
---
{query}
---

Your response should be helpful but brief (maximum 3 sentences).
"""

    def __init__(self, 
                 assessments_path: str = "shl_assessments.json", 
                 api_key: Optional[str] = None,
                 use_gemini: bool = True):
        """
        Initialize SHL Recommender
        
        Args:
            assessments_path: Path to JSON file containing SHL assessments
            api_key: Google API key for Gemini API (if None, will try to load from env)
            use_gemini: Whether to use Gemini API for recommendations
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.use_gemini = use_gemini
        
        # Initialize Gemini client if API key is available
        if self.use_gemini and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {str(e)}")
                self.use_gemini = False
        else:
            self.use_gemini = False
            if use_gemini:
                logger.warning("No API key provided, will use fallback recommendation method")
        
        # Load assessments
        try:
            self.load_assessments(assessments_path)
        except Exception as e:
            logger.error(f"Error loading assessments: {str(e)}")
            # Create empty assessments - will need to load later
            self.assessments = []
            self.assessment_texts = []
            self.vectorizer = None
            self.tfidf_matrix = None
        
    def load_assessments(self, assessments_path: str) -> None:
        """
        Load assessments from JSON file
        
        Args:
            assessments_path: Path to JSON file containing SHL assessments
        """
        try:
            if os.path.exists(assessments_path):
                with open(assessments_path, 'r', encoding='utf-8') as f:
                    self.assessments = json.load(f)
                logger.info(f"Loaded {len(self.assessments)} assessments from {assessments_path}")
                
                # Check for required fields
                if self.assessments and all(isinstance(a, dict) and 'name' in a and 'url' in a for a in self.assessments):
                    # Create text for each assessment for TF-IDF vectorization
                    self.assessment_texts = []
                    for assessment in self.assessments:
                        text = assessment['name']
                        if 'test_type' in assessment and assessment['test_type']:
                            if isinstance(assessment['test_type'], list):
                                text += " " + " ".join(assessment['test_type'])
                            else:
                                text += " " + assessment['test_type']
                        self.assessment_texts.append(text)
                    
                    # Create TF-IDF vectorizer and matrix
                    self.vectorizer = TfidfVectorizer(stop_words='english')
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.assessment_texts)
                    logger.info("Created TF-IDF matrix for text similarity matching")
                else:
                    logger.error(f"Invalid assessment data format in {assessments_path}")
                    self.assessments = []
                    self.assessment_texts = []
                    self.vectorizer = None
                    self.tfidf_matrix = None
            else:
                logger.warning(f"Assessments file not found: {assessments_path}")
                self.assessments = []
                self.assessment_texts = []
                self.vectorizer = None
                self.tfidf_matrix = None
        except Exception as e:
            logger.error(f"Error loading assessments from {assessments_path}: {str(e)}")
            raise
    
    def clean_gemini_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Clean JSON from Gemini response
        
        Args:
            response: Raw response from Gemini
            
        Returns:
            List of recommendation dictionaries
        """
        # Extract JSON string from response if needed
        json_match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response.strip()
        
        try:
            # Clean any potential markdown or extra characters
            json_str = re.sub(r'^[^[{]*', '', json_str)
            json_str = re.sub(r'[^}\]]*$', '', json_str)
            
            # Parse JSON
            recommendations = json.loads(json_str)
            return recommendations
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from Gemini response: {response}")
            return []
    
    async def get_recommendations_gemini(self, query: str, num_recommendations: int = 3) -> List[RecommendationResult]:
        """
        Get recommendations using Gemini API
        
        Args:
            query: Query string or job description
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of RecommendationResult objects
        """
        if not self.use_gemini or not self.api_key:
            logger.warning("Gemini API not configured, falling back to text similarity")
            return await self.get_recommendations_similarity(query, num_recommendations)
        
        if not self.assessments:
            logger.warning("No assessments loaded, cannot generate recommendations")
            return []
        
        try:
            # Prepare assessment info
            assessment_info = []
            for assessment in self.assessments:
                info = {
                    "name": assessment["name"],
                    "remote_testing": assessment.get("remote_testing", False),
                    "adaptive_irt": assessment.get("adaptive_irt", False),
                    "test_type": assessment.get("test_type", [])
                }
                if assessment.get("duration") is not None:
                    info["duration"] = assessment["duration"]
                assessment_info.append(info)
            
            assessment_info_str = json.dumps(assessment_info, indent=2)
            
            # Generate prompt
            prompt = self.PROMPT_TEMPLATE.format(
                query=query,
                assessment_info=assessment_info_str
            )
            
            # Call Gemini API
            response = await self.model.generate_content_async(
                self.SYSTEM_PROMPT + "\n" + prompt, 
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json"
                }
            )
            
            response_text = response.text
            logger.debug(f"Gemini response: {response_text}")
            
            # Parse response
            recommendations = self.clean_gemini_response(response_text)
            
            # Validate and convert to RecommendationResult objects
            results = []
            for rec in recommendations:
                assessment_name = rec.get("assessment_name")
                if not assessment_name:
                    continue
                
                # Find matching assessment in our list
                matching_assessment = None
                for assessment in self.assessments:
                    if assessment["name"].lower() == assessment_name.lower():
                        matching_assessment = assessment
                        break
                
                if not matching_assessment:
                    continue
                
                # Create RecommendationResult
                result = RecommendationResult(
                    assessment_name=matching_assessment["name"],
                    assessment_url=matching_assessment["url"],
                    relevance_score=float(rec.get("relevance_score", 0.5)),
                    explanation=rec.get("explanation"),
                    duration=matching_assessment.get("duration"),
                    remote_testing=matching_assessment.get("remote_testing", False),
                    adaptive_irt=matching_assessment.get("adaptive_irt", False),
                    test_type=matching_assessment.get("test_type", [])
                )
                results.append(result)
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit to requested number
            results = results[:num_recommendations]
            
            # If we don't have enough results, supplement with similarity-based recommendations
            if len(results) < num_recommendations:
                logger.info(f"Got {len(results)} from Gemini, supplementing with similarity-based recommendations")
                sim_recommendations = await self.get_recommendations_similarity(query, num_recommendations + len(results))
                
                # Filter out recommendations we already have
                existing_names = {r.assessment_name for r in results}
                sim_recommendations = [r for r in sim_recommendations if r.assessment_name not in existing_names]
                
                # Add to results
                results.extend(sim_recommendations[:num_recommendations - len(results)])
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting recommendations from Gemini: {str(e)}")
            # Fall back to similarity-based recommendations
            return await self.get_recommendations_similarity(query, num_recommendations)
    
    async def get_recommendations_similarity(self, query: str, num_recommendations: int = 3) -> List[RecommendationResult]:
        """
        Get recommendations based on text similarity
        
        Args:
            query: Query string or job description
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of RecommendationResult objects
        """
        if not self.assessments or not self.vectorizer or self.tfidf_matrix is None:
            logger.warning("No assessments loaded or vectorizer not initialized")
            return []
        
        try:
            # Transform query using the same vectorizer
            query_vec = self.vectorizer.transform([query])
            
            # Calculate similarity scores
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get indices of top matches
            top_indices = sim_scores.argsort()[-num_recommendations*2:][::-1]
            
            # Create recommendation results
            results = []
            for i in top_indices:
                if i < len(self.assessments):
                    assessment = self.assessments[i]
                    
                    # Create recommendation result
                    result = RecommendationResult(
                        assessment_name=assessment["name"],
                        assessment_url=assessment["url"],
                        relevance_score=float(sim_scores[i]),
                        duration=assessment.get("duration"),
                        remote_testing=assessment.get("remote_testing", False),
                        adaptive_irt=assessment.get("adaptive_irt", False),
                        test_type=assessment.get("test_type", [])
                    )
                    results.append(result)
            
            # Limit to requested number and return
            return results[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting recommendations using similarity: {str(e)}")
            return []
    
    def generate_explanation(self, assessment_name: str, query: str) -> str:
        """
        Generate explanation for why an assessment is recommended
        
        Args:
            assessment_name: Name of the assessment
            query: Original query
            
        Returns:
            Explanation string
        """
        if not self.use_gemini or not self.api_key:
            return ""
        
        try:
            prompt = f"""As an expert in SHL assessments, explain in 1-2 sentences why the "{assessment_name}" 
            assessment would be appropriate for this job or query: "{query}"."""
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 100
                }
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return ""
    
    async def get_recommendations(self, query: str, num_recommendations: int = 3) -> List[RecommendationResult]:
        """
        Get recommendations for a query
        
        Args:
            query: Query string or job description
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of RecommendationResult objects
        """
        # Check if we have assessments
        if not self.assessments:
            logger.warning("No assessments loaded, cannot generate recommendations")
            return []
        
        # Use Gemini if available
        if self.use_gemini and self.api_key:
            return await self.get_recommendations_gemini(query, num_recommendations)
        else:
            return await self.get_recommendations_similarity(query, num_recommendations)
    
    async def generate_generic_response(self, query: str) -> str:
        """
        Generate a generic response for queries that don't match assessments
        
        Args:
            query: User query
            
        Returns:
            Response string
        """
        if not self.use_gemini or not self.api_key:
            return "I'm unable to provide specific information about that query. Try asking about specific assessment types or job roles."
        
        try:
            prompt = self.FALLBACK_PROMPT.format(query=query)
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 150
                }
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating generic response: {str(e)}")
            return "I'm unable to provide specific information about that query. Try asking about specific assessment types or job roles."

# Singleton instance for the application
recommender_instance = None

def get_recommender(
    assessments_path: str = "shl_assessments.json",
    api_key: Optional[str] = None,
    use_gemini: bool = True
) -> SHLRecommender:
    """
    Get or create the recommender instance
    
    Args:
        assessments_path: Path to JSON file containing SHL assessments
        api_key: Google API key for Gemini API
        use_gemini: Whether to use Gemini API for recommendations
        
    Returns:
        SHLRecommender instance
    """
    global recommender_instance
    
    if recommender_instance is None:
        try:
            recommender_instance = SHLRecommender(
                assessments_path=assessments_path,
                api_key=api_key,
                use_gemini=use_gemini
            )
        except Exception as e:
            logger.error(f"Error creating recommender: {str(e)}")
            # Create a minimal recommender without assessments
            recommender_instance = SHLRecommender(
                assessments_path="",
                api_key=api_key,
                use_gemini=use_gemini
            )
    
    return recommender_instance

async def get_recommendations(query: str, num_recommendations: int = 3) -> List[RecommendationResult]:
    """
    Get recommendations for a query
    
    Args:
        query: Query string or job description
        num_recommendations: Number of recommendations to return
        
    Returns:
        List of RecommendationResult objects
    """
    recommender = get_recommender()
    return await recommender.get_recommendations(query, num_recommendations)

async def main():
    """Main function to test the recommender"""
    import asyncio
    from dotenv import load_dotenv
    
    # Load API key from .env file
    load_dotenv()
    
    # Test queries
    test_queries = [
        "Software Engineer with Java experience",
        "Marketing Manager with digital marketing background",
        "Data Scientist with Python and machine learning skills",
        "Customer Service Representative for a call center",
        "Sales Executive for enterprise software products"
    ]
    
    # Initialize recommender
    recommender = get_recommender()
    
    # Get recommendations for each query
    for query in test_queries:
        print(f"\nQuery: {query}")
        recommendations = await recommender.get_recommendations(query, num_recommendations=3)
        
        if recommendations:
            print("Recommended assessments:")
            for i, rec in enumerate(recommendations):
                print(f"{i+1}. {rec.assessment_name}")
                print(f"   Relevance: {rec.relevance_score:.2f}")
                print(f"   Duration: {rec.duration} minutes")
                print(f"   Remote: {rec.remote_testing}, Adaptive: {rec.adaptive_irt}")
                if rec.explanation:
                    print(f"   Explanation: {rec.explanation}")
                print(f"   URL: {rec.assessment_url}")
        else:
            print("No recommendations found.")

if __name__ == "__main__":
    asyncio.run(main()) 