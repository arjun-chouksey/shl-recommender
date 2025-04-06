# SHL Assessment Recommender

An intelligent recommendation system that assists hiring managers in finding relevant SHL assessments based on natural language queries or job descriptions.

## Features

- **Semantic Search**: Find assessments that match your needs using natural language processing
- **Gemini AI Integration**: Powered by Google's Gemini AI for context-aware recommendations
- **Rich Web Interface**: User-friendly Streamlit UI for exploring recommendations
- **RESTful API**: Programmatic access to recommendations via FastAPI
- **Evaluation Metrics**: Built-in evaluation of recommendation quality (MAP@K, Recall@K)
- **Robust Scraper**: Up-to-date assessment data from SHL's product catalog

## System Architecture

The system consists of three main components:

1. **Scraper**: Extracts assessment data from SHL's product catalog
2. **Recommendation Engine**: Matches job descriptions to assessments using semantic search and Gemini AI
3. **Web Interface**: Streamlit app for interacting with the recommendations

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/shl-recommender.git
   cd shl-recommender
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Playwright browsers (for scraping):
   ```bash
   playwright install
   ```

5. Set up Google Gemini API key:
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root with the following content:
     ```
     GOOGLE_API_KEY=your_api_key_here
     API_URL=http://localhost:8000
     ```

## Usage

### Running the System

You can run various components independently or all together:

```bash
# Run the scraper to get the latest assessment data
python main.py --scrape

# Run the API server only
python main.py --api

# Run the web app only
python main.py --app

# Run both API and web app
python main.py --all

# Run evaluation to measure recommendation quality
python main.py --evaluate

# Run a test query
python main.py --query "Software Engineer with Java experience"
```

### Accessing the Application

- Web UI: Open `http://localhost:8501` in your browser
- API documentation: Open `http://localhost:8000/docs` in your browser

### API Usage

#### GET Request

```bash
curl -X 'GET' \
  'http://localhost:8000/recommendations?query=Software%20Engineer%20with%20Java%20experience&num_recommendations=3' \
  -H 'accept: application/json'
```

#### POST Request

```bash
curl -X 'POST' \
  'http://localhost:8000/recommendations' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Software Engineer with Java experience",
  "num_recommendations": 3
}'
```

## Components

The system consists of several key files:

- `scraper.py`: Web scraper for SHL's assessment catalog
- `recommender.py`: The core recommendation engine
- `evaluation.py`: Metrics and benchmarking for recommendations
- `api.py`: FastAPI implementation for the backend
- `app.py`: Streamlit web interface
- `main.py`: CLI entry point to run various components

## Example Queries

- "Software Engineer with Java and Spring experience"
- "Marketing Manager with digital marketing background"
- "Data Scientist with machine learning expertise"
- "Customer Service Representative for a call center"
- "Project Manager for software development team"

## License

This project is licensed under the MIT License - see the LICENSE file for details. 