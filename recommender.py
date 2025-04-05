import json
import numpy as np
import pandas as pd
# Removed: from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
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

# Define assessment schema
class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: bool
    adaptive_irt: bool
    duration: Optional[int] = None  # in minutes
    test_type: List[str] = Field(default_factory=list)

class SHLRecommender:
    def __init__(self, data_path: str = "shl_assessments.json"):
        """
        Initialize the SHL Recommender
        
        Args:
            data_path: Path to the JSON file containing assessment data
        """
        self.assessments = self._load_assessments(data_path)
        
        # For vector similarity search
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.assessment_vectors = None
        
        # For keyword matching
        self.keyword_index = {}
        
        # Build indices
        self._build_indices()
    
    def _load_assessments(self, data_path: str) -> List[Assessment]:
        """Load assessments from a JSON file"""
        if not os.path.exists(data_path):
            logger.warning(f"Assessment data file not found at {data_path}. Using empty list.")
            return []
        
        try:
            with open(data_path, 'r') as f:
                assessment_dicts = json.load(f)
            
            return [Assessment(**assessment) for assessment in assessment_dicts]
        except Exception as e:
            logger.error(f"Error loading assessments: {str(e)}")
            return []
    
    def _build_indices(self):
        """Build indices for efficient recommendation"""
        if not self.assessments:
            logger.warning("No assessments to build indices for")
            return
        
        logger.info(f"Building indices for {len(self.assessments)} assessments")
        
        # Prepare documents for TF-IDF vectorization
        docs = []
        
        for i, assessment in enumerate(self.assessments):
            # Create document text
            doc_text = f"{assessment.name} {' '.join(assessment.test_type)}"
            docs.append(doc_text)
            
            # Build keyword index
            self._add_to_keyword_index(i, assessment)
        
        # Create TF-IDF vectors
        self.assessment_vectors = self.vectorizer.fit_transform(docs)
        
        logger.info("Indices built successfully")
    
    def _add_to_keyword_index(self, idx: int, assessment: Assessment):
        """Add an assessment to the keyword index"""
        # Extract keywords from name
        name_keywords = re.findall(r'\w+', assessment.name.lower())
        
        # Add assessment name keywords to index
        for keyword in name_keywords:
            if len(keyword) > 2:  # Skip very short words
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                if idx not in self.keyword_index[keyword]:
                    self.keyword_index[keyword].append(idx)
        
        # Add test types to index
        for test_type in assessment.test_type:
            keyword = test_type.lower()
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            if idx not in self.keyword_index[keyword]:
                self.keyword_index[keyword].append(idx)
            
            # Also add individual words from test type
            type_keywords = re.findall(r'\w+', test_type.lower())
            for type_keyword in type_keywords:
                if len(type_keyword) > 2:  # Skip very short words
                    if type_keyword not in self.keyword_index:
                        self.keyword_index[type_keyword] = []
                    if idx not in self.keyword_index[type_keyword]:
                        self.keyword_index[type_keyword].append(idx)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from a query string"""
        # Simple extraction of words, could be improved with NLP
        keywords = re.findall(r'\w+', query.lower())
        return [k for k in keywords if len(k) > 2]  # Skip very short words
    
    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Union[int, float]]]:
        """
        Search using keyword matching
        
        Returns:
            List of dicts with assessment index and score
        """
        keywords = self._extract_keywords(query)
        
        # Count keyword matches for each assessment
        assessment_scores = {}
        
        for keyword in keywords:
            if keyword in self.keyword_index:
                for idx in self.keyword_index[keyword]:
                    if idx not in assessment_scores:
                        assessment_scores[idx] = 0
                    assessment_scores[idx] += 1
        
        # Normalize scores by the number of keywords
        results = []
        for idx, score in assessment_scores.items():
            results.append({
                "idx": idx,
                "score": score / max(1, len(keywords))
            })
        
        # Sort by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def _vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Union[int, float]]]:
        """
        Search using vector similarity
        
        Returns:
            List of dicts with assessment index and score
        """
        if not self.assessments:
            return []
        
        # Transform query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity with all assessments
        similarities = cosine_similarity(query_vector, self.assessment_vectors)[0]
        
        # Create results
        results = []
        for idx, score in enumerate(similarities):
            results.append({
                "idx": idx,
                "score": float(score)
            })
        
        # Sort by score in descending order
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def _gemini_classify(self, query: str, assessment: Assessment) -> float:
        """
        Use Gemini to classify how well an assessment matches a query
        
        Returns:
            Score between 0 and 1
        """
        if not GOOGLE_API_KEY:
            return 0.0
        
        try:
            # Build prompt
            prompt = f"""
            Rate how well the following SHL assessment matches the user's requirements.
            
            User query: {query}
            
            Assessment details:
            - Name: {assessment.name}
            - Test types: {', '.join(assessment.test_type)}
            - Remote testing: {assessment.remote_testing}
            - Adaptive testing: {assessment.adaptive_irt}
            - Duration: {assessment.duration if assessment.duration else 'Unknown'} minutes
            
            Output just a number between 0 and 1, where 1 means perfect match and 0 means not relevant at all.
            """
            
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            # Parse score from response
            score_text = response.text.strip()
            # Extract a float from the response
            match = re.search(r'(\d+\.\d+|\d+)', score_text)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
            
            return 0.0
        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")
            return 0.0
    
    def _extract_duration_limit(self, query: str) -> Optional[int]:
        """Extract duration limit from query"""
        # Look for patterns like "under 30 minutes", "less than 25 min", etc.
        patterns = [
            r'under (\d+) min',
            r'less than (\d+) min',
            r'no more than (\d+) min',
            r'maximum (\d+) min',
            r'max (\d+) min',
            r'(\d+) min or less',
            r'(\d+) minutes or less',
            r'(\d+) min max',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def recommend(self, query: str, max_duration: Optional[int] = None, top_k: int = 10) -> List[Dict]:
        """
        Recommend SHL assessments based on a query
        
        Args:
            query: User query
            max_duration: Maximum duration in minutes (optional)
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended assessments with scores
        """
        if not self.assessments:
            return []
        
        logger.info(f"Generating recommendations for query: '{query}'")
        
        # Extract duration limit from query if not provided
        if max_duration is None:
            max_duration = self._extract_duration_limit(query)
        
        start_time = time.time()
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, top_k=top_k*2)
        
        # Get vector search results
        vector_results = self._vector_search(query, top_k=top_k*2)
        
        # Combine results
        combined_scores = {}
        
        # Add keyword scores (weight: 0.3)
        for result in keyword_results:
            idx = result["idx"]
            if idx not in combined_scores:
                combined_scores[idx] = 0
            combined_scores[idx] += result["score"] * 0.3
        
        # Add vector scores (weight: 0.7)
        for result in vector_results:
            idx = result["idx"]
            if idx not in combined_scores:
                combined_scores[idx] = 0
            combined_scores[idx] += result["score"] * 0.7
        
        # Create combined results
        results = []
        for idx, score in combined_scores.items():
            assessment = self.assessments[idx]
            
            # Skip if duration exceeds max_duration
            if max_duration is not None and assessment.duration is not None and assessment.duration > max_duration:
                continue
            
            results.append({
                "assessment": assessment,
                "score": score
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Get top results
        top_results = results[:top_k*2]
        
        # Rerank with Gemini if available
        if GOOGLE_API_KEY:
            for result in top_results:
                gemini_score = self._gemini_classify(query, result["assessment"])
                # Weight: 0.6 original score, 0.4 Gemini score
                result["score"] = result["score"] * 0.6 + gemini_score * 0.4
            
            # Resort
            top_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Format final results
        final_results = []
        for i, result in enumerate(top_results[:top_k]):
            assessment = result["assessment"]
            final_results.append({
                "name": assessment.name,
                "url": assessment.url,
                "score": round(result["score"], 3),
                "test_type": assessment.test_type,
                "remote_testing": assessment.remote_testing,
                "adaptive_irt": assessment.adaptive_irt,
                "duration": assessment.duration
            })
        
        elapsed_time = time.time() - start_time
        logger.info(f"Recommendations generated in {elapsed_time:.2f} seconds")
        
        return final_results


def create_sample_data():
    """Create sample assessment data if the real data doesn't exist"""
    if os.path.exists("shl_assessments.json"):
        return
    
    sample_assessments = [
        {
            "name": "Verify Numerical Reasoning Test",
            "url": "https://example.com/verify-numerical",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 25,
            "test_type": ["Numerical Reasoning", "Cognitive Ability"]
        },
        {
            "name": "Verify Verbal Reasoning Test",
            "url": "https://example.com/verify-verbal",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 25,
            "test_type": ["Verbal Reasoning", "Cognitive Ability"]
        },
        {
            "name": "Verify General Ability Test",
            "url": "https://example.com/verify-ga",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 36,
            "test_type": ["Numerical Reasoning", "Verbal Reasoning", "Inductive Reasoning", "Cognitive Ability"]
        },
        {
            "name": "Work Strengths Questionnaire",
            "url": "https://example.com/work-strengths",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": 25,
            "test_type": ["Personality Assessment"]
        },
        {
            "name": "ADEPT-15 Personality Assessment",
            "url": "https://example.com/adept15",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": 25,
            "test_type": ["Personality Assessment"]
        }
    ]
    
    # Save sample data
    with open("sample_assessments.json", "w") as f:
        json.dump(sample_assessments, f, indent=2)
    
    logger.info("Created sample assessment data in sample_assessments.json")


def main():
    """Main function for testing"""
    create_sample_data()
    
    # Initialize recommender with sample data if real data doesn't exist
    data_path = "shl_assessments.json" if os.path.exists("shl_assessments.json") else "sample_assessments.json"
    recommender = SHLRecommender(data_path=data_path)
    
    # Test queries
    test_queries = [
        "I need a test for analytical thinking",
        "Looking for a personality assessment for leadership roles",
        "Numerical reasoning test that can be taken remotely",
        "Short cognitive assessment under 30 minutes for technical roles"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = recommender.recommend(query, top_k=2)
        for i, result in enumerate(results):
            print(f"{i+1}. {result['name']} (Score: {result['score']})")
            print(f"   Type: {', '.join(result['test_type'])}")
            print(f"   Remote: {result['remote_testing']}, Duration: {result['duration']} min")


if __name__ == "__main__":
    main() 