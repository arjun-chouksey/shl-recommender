import os
import json
import argparse
import threading
import subprocess
import time
from scraper import SHLScraper

def run_scraper(force=False):
    """Run the scraper to get assessment data"""
    if force or not os.path.exists("shl_assessments.json") or os.path.getsize("shl_assessments.json") <= 5:
        print("Running scraper to fetch assessment data...")
        scraper = SHLScraper()
        assessments = scraper.scrape_catalog()
        scraper.save_to_json()
        print(f"Scraper completed! Found {len(assessments)} assessments.")
        return True
    return False

def run_api():
    """Run the FastAPI backend"""
    from api import app
    import uvicorn
    print("Starting API server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_app():
    """Run the Streamlit web app"""
    print("Starting Streamlit web app on port 8501...")
    os.system("streamlit run app.py")

def run_both():
    """Run both API and web app in separate threads"""
    api_thread = threading.Thread(target=run_api)
    app_thread = threading.Thread(target=run_app)
    
    api_thread.start()
    time.sleep(2)  # Give API time to start
    app_thread.start()
    
    api_thread.join()
    app_thread.join()

def main():
    parser = argparse.ArgumentParser(description="SHL Assessment Recommender")
    parser.add_argument("--scrape", action="store_true", help="Run scraper only")
    parser.add_argument("--api", action="store_true", help="Run API only")
    parser.add_argument("--app", action="store_true", help="Run web app only")
    parser.add_argument("--force-scrape", action="store_true", help="Force re-scrape even if data exists")
    
    args = parser.parse_args()
    
    # Run scraper if requested or if data doesn't exist
    if args.scrape or args.force_scrape:
        run_scraper(force=args.force_scrape)
        return
    
    # Check if we need to run the scraper
    if not os.path.exists("shl_assessments.json") or os.path.getsize("shl_assessments.json") <= 5:
        should_scrape = input("No assessment data found. Run scraper first? (y/n): ")
        if should_scrape.lower() == "y":
            run_scraper()
        else:
            # Create empty assessment file
            with open("shl_assessments.json", "w") as f:
                json.dump([], f)
    
    # Run API or web app if specified
    if args.api:
        run_api()
    elif args.app:
        run_app()
    else:
        # Run both by default
        run_both()

if __name__ == "__main__":
    main() 