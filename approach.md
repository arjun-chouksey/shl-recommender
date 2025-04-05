# SHL Assessment Recommender: Implementation Approach

## Problem Statement

The challenge was to build an intelligent recommendation system that helps hiring managers find the right SHL assessments for their roles. The existing system relies on keyword searches and filters, making the process time-consuming and inefficient.

## Solution Architecture

Our solution comprises three main components:

1. **Data Collection**: A web scraper to extract assessment data from SHL's product catalog
2. **Recommendation Engine**: A semantic search system using vector embeddings
3. **User Interface**: A web application with both UI and API endpoints

## Technical Implementation

### 1. Data Collection

We implemented a web scraper using **BeautifulSoup4** and **Requests** to extract assessment data from SHL's product catalog. The scraper:

- Navigates to the SHL product catalog page
- Extracts assessment details from the tables (name, URL, remote testing, adaptive/IRT, test type)
- Visits individual assessment pages to extract duration information
- Saves the data to JSON and CSV formats for accessibility

### 2. Recommendation Engine

The core of our system is a semantic search engine built with:

- **Sentence Transformers**: For creating dense vector embeddings of assessment descriptions and queries
- **FAISS**: An efficient similarity search library for fast retrieval of relevant assessments

The recommendation process follows these steps:

1. Encode all assessment descriptions into vector embeddings during initialization
2. Parse a natural language query to extract parameters (duration limits, remote testing, etc.)
3. Encode the query into a vector embedding
4. Perform a similarity search to find assessments with vectors close to the query
5. Apply filters based on the extracted parameters
6. Return the top-k most relevant assessments

### 3. User Interface

We developed two interfaces:

- **Web Application**: Built with Streamlit for an interactive user experience
- **RESTful API**: Implemented with FastAPI for programmatic access

### 4. Evaluation System

To measure the performance of our recommendation system, we implemented an evaluation module that calculates:

- **Mean Recall@K**: Measures how many of the relevant assessments were retrieved in the top K recommendations
- **MAP@K**: Evaluates both the relevance and ranking order of retrieved assessments

## Tools and Libraries Used

- **Python**: Core programming language
- **BeautifulSoup4 & Requests**: Web scraping
- **Sentence Transformers**: Vector embeddings for text
- **FAISS**: Efficient similarity search
- **FastAPI**: High-performance API framework
- **Streamlit**: Interactive web application
- **NumPy & Pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities

## Optimization Techniques

1. **Vector Indexing**: Using FAISS for efficient similarity search, which is much faster than brute-force methods
2. **Query Parameter Extraction**: Automatically extracting parameters like duration limits from natural language queries
3. **Caching**: Utilizing Streamlit's caching mechanism to avoid reloading the model and data

## Potential Improvements

1. **Fine-tuning**: Fine-tune the embedding model on SHL-specific language and assessment descriptions
2. **Job Description Analysis**: Use more sophisticated techniques to extract skills and requirements from job descriptions
3. **User Feedback Loop**: Incorporate user feedback to improve recommendations over time
4. **Hybrid Search**: Combine semantic search with keyword-based search for better precision

## Deployment Considerations

The system is designed to be easily deployable:

- Containerization with Docker for consistent execution environments
- Separation of components allows for flexible scaling options
- API documentation via FastAPI's automatic Swagger UI
- Stateless architecture for easy horizontal scaling

## Conclusion

Our SHL Assessment Recommender demonstrates how semantic search technology can be leveraged to create an intelligent recommendation system that simplifies the assessment selection process for hiring managers. 