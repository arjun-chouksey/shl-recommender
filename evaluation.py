import json
import os
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
import pandas as pd
from tabulate import tabulate
from datetime import datetime

from recommender import get_recommender, RecommendationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SHLEvaluator:
    """
    Evaluator for SHL recommendation system
    
    This class evaluates the recommendation system using standard IR metrics:
    - Mean Average Precision @ K (MAP@K)
    - Recall @ K
    - Mean Reciprocal Rank (MRR)
    - Hit Rate @ K
    """
    
    def __init__(self, queries_path: str = "benchmark_queries.json"):
        """
        Initialize the evaluator
        
        Args:
            queries_path: Path to benchmark queries JSON file
        """
        self.queries_path = queries_path
        self.queries = []
        
        # Load queries if file exists
        if os.path.exists(queries_path):
            self._load_queries()
        else:
            logger.warning(f"Benchmark queries file not found: {queries_path}")
            # Create sample benchmark queries
            self._create_sample_queries()
            
    def _create_sample_queries(self):
        """Create sample benchmark queries"""
        sample_queries = [
            {
                "query": "Frontend Developer with React experience",
                "relevant_assessments": [
                    "Verify Coding",
                    "JavaScript Programming Skills",
                    "Verify Cognitive Ability Assessment",
                    "Problem Solving Assessment"
                ]
            },
            {
                "query": "Data Scientist with machine learning experience",
                "relevant_assessments": [
                    "Python Programming Skills",
                    "Data Science Assessment",
                    "Numerical Reasoning Test",
                    "Problem Solving Assessment"
                ]
            },
            {
                "query": "Project Manager for software development team",
                "relevant_assessments": [
                    "Project Management Assessment",
                    "Leadership Assessment",
                    "ADEPT-15 Personality Assessment",
                    "Situational Judgment Test"
                ]
            },
            {
                "query": "Sales Executive for enterprise software",
                "relevant_assessments": [
                    "Sales Assessment",
                    "ADEPT-15 Personality Assessment",
                    "Emotional Intelligence Assessment",
                    "Situational Judgment Test"
                ]
            },
            {
                "query": "Customer Service Representative for call center",
                "relevant_assessments": [
                    "Customer Service Assessment",
                    "Work Strengths Questionnaire",
                    "Situational Judgment Test",
                    "Verbal Reasoning Test"
                ]
            }
        ]
        
        self.queries = sample_queries
        
        # Save sample queries
        with open(self.queries_path, 'w', encoding='utf-8') as f:
            json.dump(sample_queries, f, indent=2)
            
        logger.info(f"Created sample benchmark queries in {self.queries_path}")
    
    def _load_queries(self):
        """Load benchmark queries from file"""
        try:
            with open(self.queries_path, 'r', encoding='utf-8') as f:
                self.queries = json.load(f)
            logger.info(f"Loaded {len(self.queries)} benchmark queries from {self.queries_path}")
        except Exception as e:
            logger.error(f"Error loading benchmark queries: {str(e)}")
            self.queries = []
    
    @staticmethod
    def calculate_average_precision(relevant: Set[str], recommended: List[str], k: int) -> float:
        """
        Calculate Average Precision @ K
        
        Args:
            relevant: Set of relevant assessment names
            recommended: List of recommended assessment names in order
            k: Number of recommendations to consider
            
        Returns:
            Average Precision @ K
        """
        if not relevant or not recommended:
            return 0.0
        
        recommended = recommended[:k]
        hits = 0
        sum_precisions = 0.0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / min(len(relevant), k) if hits > 0 else 0.0
    
    @staticmethod
    def calculate_recall(relevant: Set[str], recommended: List[str], k: int) -> float:
        """
        Calculate Recall @ K
        
        Args:
            relevant: Set of relevant assessment names
            recommended: List of recommended assessment names in order
            k: Number of recommendations to consider
            
        Returns:
            Recall @ K
        """
        if not relevant:
            return 0.0
        
        recommended_set = set(recommended[:k])
        return len(recommended_set.intersection(relevant)) / len(relevant)
    
    @staticmethod
    def calculate_mrr(relevant: Set[str], recommended: List[str], k: int) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            relevant: Set of relevant assessment names
            recommended: List of recommended assessment names in order
            k: Number of recommendations to consider
            
        Returns:
            Mean Reciprocal Rank
        """
        if not relevant or not recommended:
            return 0.0
        
        recommended = recommended[:k]
        
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def calculate_hit_rate(relevant: Set[str], recommended: List[str], k: int) -> float:
        """
        Calculate Hit Rate @ K
        
        Args:
            relevant: Set of relevant assessment names
            recommended: List of recommended assessment names in order
            k: Number of recommendations to consider
            
        Returns:
            Hit Rate @ K (1.0 if there's at least one hit, 0.0 otherwise)
        """
        if not relevant or not recommended:
            return 0.0
        
        recommended_set = set(recommended[:k])
        return 1.0 if recommended_set.intersection(relevant) else 0.0
    
    async def evaluate_query(self, 
                            query: str, 
                            relevant_assessments: List[str], 
                            k_values: List[int] = [1, 3, 5]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate a single query
        
        Args:
            query: Query string
            relevant_assessments: List of relevant assessment names
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with metrics for each k value
        """
        # Get recommendations
        recommender = get_recommender()
        max_k = max(k_values)
        recommendations = await recommender.get_recommendations(query, num_recommendations=max_k)
        
        # Extract assessment names
        recommended_names = [rec.assessment_name for rec in recommendations]
        relevant_set = set(relevant_assessments)
        
        # Calculate metrics for each k
        results = {
            "MAP": {},
            "Recall": {},
            "MRR": {},
            "HitRate": {}
        }
        
        for k in k_values:
            if k > len(recommended_names):
                continue
                
            results["MAP"][k] = self.calculate_average_precision(relevant_set, recommended_names, k)
            results["Recall"][k] = self.calculate_recall(relevant_set, recommended_names, k)
            results["MRR"][k] = self.calculate_mrr(relevant_set, recommended_names, k)
            results["HitRate"][k] = self.calculate_hit_rate(relevant_set, recommended_names, k)
        
        return results
    
    async def evaluate_all(self, k_values: List[int] = [1, 3, 5], save_results: bool = True) -> Tuple[Dict[str, Dict[int, float]], List[Dict[str, Any]]]:
        """
        Evaluate all benchmark queries
        
        Args:
            k_values: List of k values to evaluate
            save_results: Whether to save results to file
            
        Returns:
            Tuple of (aggregate metrics, per-query results)
        """
        if not self.queries:
            logger.warning("No benchmark queries to evaluate")
            return {}, []
        
        all_results = []
        
        # Evaluate each query
        for query_data in self.queries:
            query = query_data["query"]
            relevant = query_data["relevant_assessments"]
            
            try:
                results = await self.evaluate_query(query, relevant, k_values)
                
                all_results.append({
                    "query": query,
                    "relevant": relevant,
                    "results": results
                })
                
                logger.info(f"Evaluated query: '{query}'")
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {str(e)}")
        
        # Calculate aggregate metrics
        agg_metrics = {
            "MAP": {k: 0.0 for k in k_values},
            "Recall": {k: 0.0 for k in k_values},
            "MRR": {k: 0.0 for k in k_values},
            "HitRate": {k: 0.0 for k in k_values}
        }
        
        if all_results:
            for k in k_values:
                map_values = [r["results"]["MAP"].get(k, 0.0) for r in all_results]
                recall_values = [r["results"]["Recall"].get(k, 0.0) for r in all_results]
                mrr_values = [r["results"]["MRR"].get(k, 0.0) for r in all_results]
                hit_rate_values = [r["results"]["HitRate"].get(k, 0.0) for r in all_results]
                
                agg_metrics["MAP"][k] = np.mean(map_values)
                agg_metrics["Recall"][k] = np.mean(recall_values)
                agg_metrics["MRR"][k] = np.mean(mrr_values)
                agg_metrics["HitRate"][k] = np.mean(hit_rate_values)
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"evaluation_results_{timestamp}.json"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": timestamp,
                    "aggregate_metrics": agg_metrics,
                    "query_results": all_results
                }, f, indent=2)
            
            logger.info(f"Saved evaluation results to {results_path}")
        
        return agg_metrics, all_results
    
    def print_evaluation_summary(self, metrics: Dict[str, Dict[int, float]], k_values: List[int] = [1, 3, 5]):
        """
        Print evaluation summary
        
        Args:
            metrics: Metrics dictionary
            k_values: List of k values to display
        """
        if not metrics:
            print("No evaluation metrics available")
            return
        
        # Create data for table
        table_data = []
        
        for metric_name in ["MAP", "Recall", "MRR", "HitRate"]:
            row = [metric_name]
            for k in k_values:
                if k in metrics[metric_name]:
                    row.append(f"{metrics[metric_name][k]:.4f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        # Create headers
        headers = ["Metric"] + [f"@{k}" for k in k_values]
        
        # Print table
        print("\nEvaluation Summary:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Check if metrics meet target
        for k in k_values:
            if k == 3:  # Check specifically for k=3
                map_k = metrics["MAP"].get(k, 0.0)
                recall_k = metrics["Recall"].get(k, 0.0)
                
                if map_k >= 0.7 and recall_k >= 0.7:
                    print(f"\nMETRIC TARGET MET! MAP@{k} = {map_k:.4f}, Recall@{k} = {recall_k:.4f}")
                else:
                    print(f"\nMETRIC TARGET NOT MET. MAP@{k} = {map_k:.4f}, Recall@{k} = {recall_k:.4f}")
                    print("Target: MAP@3 >= 0.7, Recall@3 >= 0.7")

async def evaluate_recommendations(k_values: List[int] = [1, 3, 5]) -> Dict[str, Dict[int, float]]:
    """
    Evaluate the recommendation system
    
    Args:
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary with metrics for each k value
    """
    evaluator = SHLEvaluator()
    metrics, _ = await evaluator.evaluate_all(k_values=k_values)
    evaluator.print_evaluation_summary(metrics, k_values)
    return metrics

async def main():
    """Main function to run evaluation"""
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed, skipping environment variable loading")
    
    # Run evaluation
    await evaluate_recommendations()

if __name__ == "__main__":
    asyncio.run(main()) 