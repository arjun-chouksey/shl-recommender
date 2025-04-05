import json
import argparse
import numpy as np
from typing import List, Dict, Any
from recommender import AssessmentRecommender

class RecommenderEvaluator:
    def __init__(self, recommender, test_queries_path=None):
        """
        Initialize the evaluator with a recommender and test queries
        
        Args:
            recommender: The recommender instance to evaluate
            test_queries_path: Path to the test queries JSON file
        """
        self.recommender = recommender
        self.test_queries = self._load_test_queries(test_queries_path)
    
    def _load_test_queries(self, path=None):
        """Load test queries from a JSON file or use default queries"""
        if path:
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading test queries: {e}")
        
        # Default test queries
        return [
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
            }
        ]
    
    def is_relevant(self, assessment: Dict[str, Any], relevant_keywords: List[str]) -> bool:
        """Check if an assessment is relevant based on keywords"""
        assessment_text = f"{assessment['name']} {assessment.get('test_type', '')}"
        return any(keyword.lower() in assessment_text.lower() for keyword in relevant_keywords)
    
    def precision_at_k(self, recommendations: List[Dict[str, Any]], relevant_keywords: List[str], k: int) -> float:
        """Calculate precision@k"""
        if not recommendations or k <= 0:
            return 0.0
        
        k = min(k, len(recommendations))
        relevant_count = sum(1 for i, rec in enumerate(recommendations[:k]) 
                            if self.is_relevant(rec, relevant_keywords))
        
        return relevant_count / k
    
    def recall_at_k(self, recommendations: List[Dict[str, Any]], relevant_keywords: List[str], k: int) -> float:
        """Calculate recall@k"""
        if not recommendations or not relevant_keywords or k <= 0:
            return 0.0
        
        k = min(k, len(recommendations))
        relevant_count = sum(1 for i, rec in enumerate(recommendations[:k]) 
                            if self.is_relevant(rec, relevant_keywords))
        
        # For recall, we estimate the total number of relevant items
        # Here we use the number of relevant keywords as a proxy
        return relevant_count / len(relevant_keywords)
    
    def average_precision(self, recommendations: List[Dict[str, Any]], relevant_keywords: List[str], k: int) -> float:
        """Calculate average precision for a single query"""
        if not recommendations or not relevant_keywords or k <= 0:
            return 0.0
        
        k = min(k, len(recommendations))
        relevant_sum = 0
        total_relevant = 0
        
        for i, rec in enumerate(recommendations[:k]):
            is_rel = self.is_relevant(rec, relevant_keywords)
            if is_rel:
                total_relevant += 1
                # Calculate precision up to this point
                precision_at_i = total_relevant / (i + 1)
                relevant_sum += precision_at_i
        
        if total_relevant == 0:
            return 0.0
        
        return relevant_sum / min(len(relevant_keywords), total_relevant)
    
    def evaluate(self, k: int = 3) -> Dict[str, float]:
        """
        Evaluate the recommender using Mean Recall@K and MAP@K
        
        Args:
            k: The K value for recall and precision
            
        Returns:
            Dictionary with evaluation metrics
        """
        recalls = []
        aps = []
        
        for query_obj in self.test_queries:
            query = query_obj["query"]
            relevant_keywords = query_obj["relevant_assessments"]
            
            # Get recommendations
            recommendations = self.recommender.recommend_from_query(query, top_k=k)
            
            # Calculate recall@k
            recall = self.recall_at_k(recommendations, relevant_keywords, k)
            recalls.append(recall)
            
            # Calculate average precision
            ap = self.average_precision(recommendations, relevant_keywords, k)
            aps.append(ap)
            
            print(f"Query: {query}")
            print(f"  Recall@{k}: {recall:.4f}")
            print(f"  AP@{k}: {ap:.4f}")
            print("  Recommendations:")
            for i, rec in enumerate(recommendations[:k]):
                print(f"    {i+1}. {rec['name']} (Relevant: {self.is_relevant(rec, relevant_keywords)})")
            print()
        
        mean_recall = np.mean(recalls)
        map_score = np.mean(aps)
        
        return {
            f"Mean_Recall@{k}": mean_recall,
            f"MAP@{k}": map_score
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate the SHL Assessment Recommender")
    parser.add_argument("--data", type=str, default="shl_assessments.json", help="Path to the assessment data")
    parser.add_argument("--test-queries", type=str, help="Path to test queries JSON file")
    parser.add_argument("--k", type=int, default=3, help="K value for evaluation metrics")
    
    args = parser.parse_args()
    
    # Initialize the recommender
    recommender = AssessmentRecommender(data_path=args.data)
    
    # Create the evaluator
    evaluator = RecommenderEvaluator(recommender, args.test_queries)
    
    # Run the evaluation
    metrics = evaluator.evaluate(k=args.k)
    
    # Print the results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 