# SHL Assessment Recommendation System

An intelligent recommendation system that helps hiring managers find the right SHL assessments for their roles. Given a natural language query or job description, the system recommends the most relevant SHL assessments.

## Features

- 🔍 **Semantic Search**: Uses a hybrid approach combining vector similarity (70%) and keyword matching (30%)
- 🤖 **Gemini AI Integration**: Leverages Google's Gemini API for embeddings and semantic understanding
- 🌐 **Rich Web Interface**: User-friendly Streamlit app with filtering options and detailed results
- 🔄 **RESTful API**: Comprehensive API endpoints for integration with other systems
- 📊 **Evaluation Metrics**: Built-in evaluation harness with Mean Recall@3 and MAP@3 metrics
- 🔄 **Robust Scraper**: Asynchronous scraper with error recovery and pagination handling

## System Architecture

The system consists of three main components:

1. **Scraper**: Extracts assessment data from SHL's product catalog
2. **Recommendation Engine**: Uses a hybrid approach to find relevant assessments
3. **Web Interface**: Provides a user-friendly interface for searching and viewing recommendations

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/shl-recommender.git
   cd shl-recommender
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Set up Gemini API:
   Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
   You can get an API key from [Google AI Studio](https://ai.google.dev/).

## Usage

### Running the Application

The easiest way to run the application is using the `run.py` script:

```bash
# Run both API and web app, creating sample data if needed
python run.py

# Run the scraper first to get real SHL assessment data
python run.py --scrape

# Run only the API server
python run.py --api-only

# Run only the Streamlit web app
python run.py --app-only
```

Once running, you can access:
- Web UI: http://localhost:8501
- API: http://localhost:8000/recommend?query=your+query+here
- API Documentation: http://localhost:8000/docs

### Using the Web Interface

1. Enter a natural language query or job description
2. Optionally set a maximum duration and number of results
3. View recommended assessments in a sortable table
4. Click on assessment links to view detailed information on SHL's website

### Using the API

#### GET /recommend

```
GET /recommend?query=Java developers with collaboration skills&max_duration=40&top_k=5
```

Parameters:
- `query` (required): Natural language query or job description
- `max_duration` (optional): Maximum assessment duration in minutes
- `top_k` (optional): Number of recommendations to return (default: 10)

#### POST /recommend

```
POST /recommend
Content-Type: application/json

{
  "query": "Java developers with collaboration skills",
  "max_duration": 40,
  "top_k": 5
}
```

## Components

### Scraper (`scraper.py`)

The scraper is designed to extract assessment data from SHL's product catalog with:
- Asynchronous HTTP requests for efficiency
- Auto pagination detection
- Error retries (3 attempts)
- JavaScript rendering fallback via Playwright if needed
- Structured data extraction and validation

### Recommender (`recommender.py`)

The recommendation engine uses a hybrid approach:
- Vector similarity search (70% weight) using Sentence Transformers
- Keyword matching (30% weight) for exact term matches
- Automatic parameter extraction from natural language queries
- Duration and test type filtering

### Evaluation (`evaluation.py`)

The evaluation module calculates:
- Mean Recall@K: Measures how many relevant assessments were retrieved
- MAP@K: Evaluates both relevance and ranking order
- Detailed per-query metrics for analysis

### API (`api.py`)

The FastAPI backend provides:
- RESTful API endpoints for recommendations
- Swagger documentation
- Structured request/response validation

### Web App (`app.py`)

The Streamlit web app includes:
- Query input via text or URL extraction
- Filtering options for duration and result count
- Sortable table of recommendations with direct links
- Detailed cards for each assessment
- Example queries for quick testing

## Example Queries

- "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes."
- "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with max duration of 60 minutes."
- "I am hiring for an analyst and want applications to be screened using Cognitive and personality tests, what options are available within 45 mins."
- "Need to assess leadership potential for a management position in less than 30 minutes."
- "Looking for a sales assessment that evaluates both knowledge and personality traits."

## License

MIT 