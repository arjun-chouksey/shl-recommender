import os
import argparse
import asyncio
import subprocess
import signal
import time
import sys

def run_api():
    """Run the FastAPI backend"""
    print("Starting API server on port 8000...")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

def run_app():
    """Run the Streamlit web app"""
    print("Starting Streamlit web app on port 8501...")
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

async def scrape_data():
    """Run the scraper to get assessment data"""
    if not os.path.exists("shl_assessments.json") or os.path.getsize("shl_assessments.json") <= 5:
        print("Running scraper to fetch assessment data...")
        try:
            from scraper import scrape_all_shl_assessments
            assessments = await scrape_all_shl_assessments()
            print(f"Scraper completed! Found {len(assessments)} assessments.")
            return True
        except Exception as e:
            print(f"Error running scraper: {str(e)}")
            print("Using sample data instead.")
            from recommender import create_sample_data
            create_sample_data()
            return False
    return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SHL Assessment Recommender")
    parser.add_argument("--scrape", action="store_true", help="Run the scraper first")
    parser.add_argument("--api-only", action="store_true", help="Run API server only")
    parser.add_argument("--app-only", action="store_true", help="Run Streamlit app only")
    
    args = parser.parse_args()
    
    # Run scraper if needed
    if args.scrape:
        asyncio.run(scrape_data())
    
    # Create sample data if needed
    if not os.path.exists("shl_assessments.json") and not os.path.exists("sample_assessments.json"):
        print("No assessment data found. Creating sample data...")
        from recommender import create_sample_data
        create_sample_data()
    
    # Start processes
    api_process = None
    app_process = None
    
    try:
        # Start API server
        if not args.app_only:
            api_process = run_api()
            time.sleep(2)  # Wait for API to start
        
        # Start Streamlit app
        if not args.api_only:
            app_process = run_app()
        
        # Print URLs
        if api_process:
            print("\nAPI is running at:")
            print("  http://localhost:8000/docs - API documentation")
            print("  http://localhost:8000/recommend?query=your+query+here - API endpoint")
        
        if app_process:
            print("\nWeb app is running at:")
            print("  http://localhost:8501 - Streamlit app")
        
        print("\nPress Ctrl+C to stop the server(s)\n")
        
        # Monitor logs
        while True:
            if api_process and api_process.poll() is not None:
                print("API server stopped unexpectedly!")
                break
            
            if app_process and app_process.poll() is not None:
                print("Streamlit app stopped unexpectedly!")
                break
            
            # Print output from processes (optional)
            if api_process:
                for line in api_process.stdout.readline(), api_process.stderr.readline():
                    if line and line.strip():
                        print(f"[API] {line.strip()}")
            
            if app_process:
                for line in app_process.stdout.readline(), app_process.stderr.readline():
                    if line and line.strip():
                        print(f"[APP] {line.strip()}")
                        
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Clean up processes
        if api_process:
            api_process.terminate()
        
        if app_process:
            app_process.terminate()

if __name__ == "__main__":
    main() 