import json
import numpy as np
from typing import Dict, List, Any, Set
import logging
import os
import pandas as pd
from pathlib import Path

from recommender import SHLRecommender, Assessment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationData:
    """Data class for evaluation data"""
    def __init__(self, query: str, relevant_assessments: List[str]):
        self.query = query
        self.relevant_assessments = relevant_assessments

def load_test_data(path: str = None) -> List[EvaluationData]:
    """Load test data from a JSON file or use default test data"""
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            return [EvaluationData(**item) for item in data]
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
    
    # Default test data if no file provided or file loading failed
    default_data = [
        {
            "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            "relevant_assessments": ["Java", "Collaboration", "Team Skills", "Programming", "Developer", "Technical"]
        },
        {
            "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes.",
            "relevant_assessments": ["Python", "SQL", "JavaScript", "Programming", "Developer", "Technical"]
        },
        {
            "query": "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",
            "relevant_assessments": ["Analyst", "Cognitive", "Personality", "Problem Solving", "Data", "Analytical"]
        },
        {
            "query": "Need to evaluate leadership potential for management positions. Assessment should be adaptive and take less than 30 minutes.",
            "relevant_assessments": ["Leadership", "Management", "Potential", "Executive", "Decision Making"]
        },
        {
            "query": "Looking for a comprehensive assessment for sales professionals that evaluates both sales knowledge and personality traits.",
            "relevant_assessments": ["Sales", "Personality", "Knowledge", "Customer", "Behavior"]
        }
    ]
    
    return [EvaluationData(**item) for item in default_data]

def is_assessment_relevant(assessment: Dict[str, Any], relevant_keywords: List[str]) -> bool:
    """Check if an assessment is relevant based on keywords"""
    # Create a text representation of the assessment
    assessment_text = f"{assessment['name'].lower()} {' '.join([t.lower() for t in assessment['test_type']])}"
    
    # Check if any relevant keyword is found in the assessment text
    for keyword in relevant_keywords:
        if keyword.lower() in assessment_text:
            return True
    
    return False

def calculate_recall_at_k(recommendations: List[Dict[str, Any]], relevant_keywords: List[str], k: int) -> float:
    """
    Calculate Recall@K
    
    Recall@K = Number of relevant assessments in top K / Total relevant assessments
    """
    if not recommendations or not relevant_keywords:
        return 0.0
    
    # Limit to top K recommendations
    recommendations = recommendations[:k]
    
    # Count relevant assessments in recommendations
    relevant_count = sum(1 for rec in recommendations if is_assessment_relevant(rec, relevant_keywords))
    
    # Calculate recall
    # For this implementation, we use the number of relevant keywords as denominator
    # In a real scenario, you'd want the actual count of all relevant items in the dataset
    return relevant_count / len(relevant_keywords)

def calculate_average_precision(recommendations: List[Dict[str, Any]], relevant_keywords: List[str], k: int) -> float:
    """
    Calculate Average Precision (AP) at K
    
    AP@K = (1/R) * sum(Precision@i * rel(i)) for i=1 to K
    where:
    - R = min(K, total relevant assessments)
    - Precision@i = Number of relevant items up to position i / i
    - rel(i) = 1 if the item at position i is relevant, 0 otherwise
    """
    if not recommendations or not relevant_keywords:
        return 0.0
    
    # Limit to top K recommendations
    recommendations = recommendations[:k]
    
    relevant_sum = 0
    relevant_count = 0
    
    for i, recommendation in enumerate(recommendations):
        is_relevant = is_assessment_relevant(recommendation, relevant_keywords)
        
        if is_relevant:
            relevant_count += 1
            # Precision at this point = relevant items so far / items seen so far
            precision_at_i = relevant_count / (i + 1)
            relevant_sum += precision_at_i
    
    # If no relevant items found, AP is 0
    if relevant_count == 0:
        return 0.0
    
    # Calculate AP
    # Normalize by min(K, number of relevant keywords)
    return relevant_sum / min(k, len(relevant_keywords))

def calculate_metrics(ground_truth: Dict[str, List[str]], predictions: Dict[str, List[Dict[str, Any]]], k: int = 3) -> Dict[str, float]:
    """
    Implements SHL's exact formulas for:
    - Mean Recall@3
    - MAP@3
    
    Args:
        ground_truth: Dictionary mapping query IDs to lists of relevant assessment keywords
        predictions: Dictionary mapping query IDs to lists of predicted assessment dictionaries
        k: K value for the metrics (default: 3)
    
    Returns:
        Dictionary with Mean Recall@K and MAP@K values
    """
    if not ground_truth or not predictions:
        return {"Mean_Recall@K": 0.0, "MAP@K": 0.0}
    
    # Ensure we have the same queries in both dictionaries
    common_queries = set(ground_truth.keys()) & set(predictions.keys())
    
    if not common_queries:
        logger.warning("No common queries between ground truth and predictions")
        return {"Mean_Recall@K": 0.0, "MAP@K": 0.0}
    
    recall_values = []
    ap_values = []
    
    for query_id in common_queries:
        relevant_keywords = ground_truth[query_id]
        recommendations = predictions[query_id]
        
        # Calculate Recall@K
        recall = calculate_recall_at_k(recommendations, relevant_keywords, k)
        recall_values.append(recall)
        
        # Calculate AP@K
        ap = calculate_average_precision(recommendations, relevant_keywords, k)
        ap_values.append(ap)
    
    # Calculate Mean Recall@K and MAP@K
    mean_recall = sum(recall_values) / len(recall_values)
    map_k = sum(ap_values) / len(ap_values)
    
    return {
        f"Mean_Recall@{k}": mean_recall,
        f"MAP@{k}": map_k
    }

def evaluate_recommender(recommender: SHLRecommender, test_data: List[EvaluationData], k: int = 3) -> Dict[str, float]:
    """
    Evaluate a recommender using the given test data
    
    Args:
        recommender: SHLRecommender instance
        test_data: List of EvaluationData objects
        k: K value for evaluation metrics
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Prepare ground truth and predictions
    ground_truth = {}
    predictions = {}
    
    # Get recommendations for each query
    for i, data in enumerate(test_data):
        query_id = f"query_{i}"
        ground_truth[query_id] = data.relevant_assessments
        
        # Get recommendations
        recommendations = recommender.recommend(data.query, top_k=k)
        predictions[query_id] = recommendations
        
        # Log results for analysis
        logger.info(f"Query: {data.query}")
        logger.info(f"  Relevant keywords: {data.relevant_assessments}")
        logger.info(f"  Recommendations: {[rec['name'] for rec in recommendations]}")
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions, k)
    
    # Log metrics
    logger.info(f"Evaluation metrics (k={k}):")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def main():
    """Main function to run the evaluation"""
    # Load test data
    test_data = load_test_data()
    
    # Find assessment data file
    if os.path.exists("shl_assessments.json"):
        data_path = "shl_assessments.json"
    elif os.path.exists("sample_assessments.json"):
        data_path = "sample_assessments.json"
    else:
        logger.error("No assessment data found. Please run scraper.py first.")
        return
    
    # Initialize recommender
    recommender = SHLRecommender(data_path=data_path)
    
    # Evaluate
    k_values = [1, 3, 5]
    results = {}
    
    for k in k_values:
        metrics = evaluate_recommender(recommender, test_data, k)
        results[k] = metrics
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 40)
    for k, metrics in results.items():
        print(f"k = {k}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Create a more detailed results dataframe
    detailed_results = []
    
    for i, data in enumerate(test_data):
        query_id = f"query_{i}"
        recommendations = recommender.recommend(data.query, top_k=max(k_values))
        
        # Calculate metrics for this specific query
        recall_values = []
        ap_values = []
        
        for k in k_values:
            recall = calculate_recall_at_k(recommendations, data.relevant_assessments, k)
            ap = calculate_average_precision(recommendations, data.relevant_assessments, k)
            
            recall_values.append(recall)
            ap_values.append(ap)
        
        # Add to detailed results
        detailed_results.append({
            "query": data.query,
            "relevant_keywords": ", ".join(data.relevant_assessments),
            "top_recommendations": ", ".join([rec["name"] for rec in recommendations[:3]]),
            **{f"Recall@{k}": recall_values[i] for i, k in enumerate(k_values)},
            **{f"AP@{k}": ap_values[i] for i, k in enumerate(k_values)}
        })
    
    # Save detailed results to CSV
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv("evaluation_results.csv", index=False)
    print(f"Detailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    main() 