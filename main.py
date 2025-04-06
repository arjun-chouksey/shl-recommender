import os
import asyncio
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_scraper(force_refresh: bool = False):
    """
    Run the SHL assessment scraper
    
    Args:
        force_refresh: Force a fresh scrape even if cached data exists
    """
    try:
        from scraper import scrape_all_shl_assessments
        logger.info("Running SHL assessment scraper...")
        assessments, freshly_scraped = await scrape_all_shl_assessments(force_refresh=force_refresh)
        logger.info(f"{'Scraped' if freshly_scraped else 'Loaded'} {len(assessments)} assessments")
        return assessments
    except ImportError:
        logger.error("Scraper module not found")
        return []
    except Exception as e:
        logger.error(f"Error running scraper: {str(e)}")
        return []

def run_api_server():
    """Run the API server"""
    try:
        from api import run_api
        logger.info("Starting API server...")
        run_api()
    except ImportError:
        logger.error("API module not found")
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")

def run_streamlit_app():
    """Run the Streamlit app"""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        logger.info("Starting Streamlit app...")
        sys.argv = ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
        stcli.main()
    except ImportError:
        logger.error("Streamlit not found")
    except Exception as e:
        logger.error(f"Error starting Streamlit app: {str(e)}")

async def run_evaluation(k_values=None):
    """
    Run evaluation on the recommendation system
    
    Args:
        k_values: List of k values to evaluate
    """
    try:
        from evaluation import evaluate_recommendations
        logger.info("Running evaluation...")
        if k_values is None:
            k_values = [1, 3, 5]
        metrics = await evaluate_recommendations(k_values)
        
        # Check if metrics meet target for k=3
        if 3 in k_values:
            map_3 = metrics["MAP"].get(3, 0.0)
            recall_3 = metrics["Recall"].get(3, 0.0)
            
            if map_3 >= 0.7 and recall_3 >= 0.7:
                logger.info(f"METRICS TARGET MET! MAP@3 = {map_3:.4f}, Recall@3 = {recall_3:.4f}")
            else:
                logger.warning(f"METRICS TARGET NOT MET. MAP@3 = {map_3:.4f}, Recall@3 = {recall_3:.4f}")
        
        return metrics
    except ImportError:
        logger.error("Evaluation module not found")
        return {}
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        return {}

async def run_demo_query(query: str, num_recommendations: int = 3):
    """
    Run a demo query against the recommender
    
    Args:
        query: Query text or job description
        num_recommendations: Number of recommendations to return
    """
    try:
        from recommender import get_recommender
        logger.info(f"Running demo query: {query}")
        
        # Get recommender instance
        recommender = get_recommender()
        
        # Get recommendations
        recommendations = await recommender.get_recommendations(query, num_recommendations)
        
        # Display recommendations
        print(f"\nQuery: {query}")
        print("Recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"\n{i+1}. {rec.assessment_name}")
            print(f"   Relevance: {rec.relevance_score:.2f}")
            if rec.explanation:
                print(f"   Explanation: {rec.explanation}")
            print(f"   Duration: {rec.duration} minutes")
            print(f"   Remote Testing: {'Yes' if rec.remote_testing else 'No'}")
            print(f"   Adaptive/IRT: {'Yes' if rec.adaptive_irt else 'No'}")
            print(f"   Test Types: {', '.join(rec.test_type)}")
            print(f"   URL: {rec.assessment_url}")
        
        return recommendations
    except ImportError:
        logger.error("Recommender module not found")
        return []
    except Exception as e:
        logger.error(f"Error running demo query: {str(e)}")
        return []

async def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SHL Assessment Recommender")
    parser.add_argument("--scrape", action="store_true", help="Scrape SHL assessments")
    parser.add_argument("--playwright", action="store_true", help="Use Playwright for scraping")
    parser.add_argument(
        "--max-depth", 
        type=int,
        default=5,
        help="Maximum depth for scraping pagination"
    )
    parser.add_argument("--query", type=str, help="Run a demo query")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--api", action="store_true", help="Run API server")
    parser.add_argument("--app", action="store_true", help="Run Streamlit app")
    parser.add_argument("--direct", action="store_true", help="Run in direct mode (no API)")
    parser.add_argument("--all", action="store_true", help="Run all components")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Scrape assessments if requested
    if args.scrape or args.all:
        from scraper import SHLScraper
        
        logger.info("Scraping SHL assessments...")
        async with SHLScraper(use_playwright=args.playwright, max_depth=args.max_depth) as scraper:
            try:
                assessments = await scraper.scrape_all_shl_assessments()
                scraper.save_to_json("shl_assessments.json")
                scraper.save_to_csv("shl_assessments.csv")
                logger.info(f"Successfully scraped {len(assessments)} assessments")
            except Exception as e:
                logger.error(f"Error scraping assessments: {str(e)}")
                if args.all:
                    logger.warning("Continuing with sample data...")
                else:
                    raise
    
    # Run query if requested
    if args.query:
        await run_demo_query(args.query)
    
    # Run evaluation if requested
    if args.evaluate:
        await run_evaluation()
    
    # Run API server if requested
    if args.api or args.all:
        # Run in a separate process if running all components
        if args.all:
            import multiprocessing
            api_process = multiprocessing.Process(target=run_api_server)
            api_process.start()
            logger.info("API server started in separate process")
        else:
            run_api_server()
    
    # Run Streamlit app if requested
    if args.app or args.all:
        run_streamlit_app()

    # Run direct mode if requested
    if args.direct:
        os.environ["DIRECT_MODE"] = "1"
        run_streamlit_app()

if __name__ == "__main__":
    asyncio.run(main()) 