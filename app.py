import streamlit as st
import requests
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from urllib.parse import urlparse
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import recommender for direct usage when API is not available
try:
    from recommender import SHLRecommender, create_sample_data
    DIRECT_MODE = True
except ImportError:
    DIRECT_MODE = False
    logger.warning("Could not import recommender module, will use API only")

# Constants
# Safely get API URL from various sources with fallbacks
API_URL = None
try:
    # Try getting from secrets
    API_URL = st.secrets["API_URL"]
    logger.info("Using API URL from Streamlit secrets")
except Exception as e:
    logger.warning(f"Could not load from secrets: {str(e)}")
    # Try getting from environment
    API_URL = os.environ.get("API_URL")
    if API_URL:
        logger.info("Using API URL from environment variables")
    else:
        # Use default as last resort
        API_URL = "https://shl-recommender-api-wypm.onrender.com"
        logger.info("Using default API URL")

logger.info(f"API URL: {API_URL}")

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066b2;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .assessment-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #0066b2;
    }
    .assessment-title {
        font-size: 1.2rem;
        color: #0066b2;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .assessment-score {
        font-size: 1rem;
        color: #198754;
        font-weight: bold;
    }
    .assessment-meta {
        color: #666;
        font-size: 0.9rem;
    }
    .footer {
        font-size: 0.8rem;
        color: #666;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    .url-info {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def get_recommender():
    """Initialize the recommender directly (when API is not available)"""
    if os.path.exists("shl_assessments.json"):
        data_path = "shl_assessments.json"
    elif os.path.exists("sample_assessments.json"):
        data_path = "sample_assessments.json"
    else:
        # Create sample data
        create_sample_data()
        data_path = "sample_assessments.json"
    
    return SHLRecommender(data_path=data_path)

def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL"""
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            st.error("Invalid URL. Please enter a valid URL.")
            return ""
        
        # Fetch content
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            
            # Extract text from HTML (simplified)
            content = response.text
            # You could use BeautifulSoup here for better extraction
            # but keeping it simple for the example
            import re
            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Limit length
            return text[:5000]
    except Exception as e:
        logger.error(f"Error extracting text from URL: {str(e)}")
        st.error(f"Error extracting text from URL: {str(e)}")
        return ""

def get_recommendations_from_api(query: str, max_duration: Optional[int] = None, top_k: int = 10) -> Dict[str, Any]:
    """Get recommendations from the API"""
    try:
        params = {
            "query": query,
            "top_k": top_k
        }
        
        if max_duration and max_duration > 0:
            params["max_duration"] = max_duration
        
        response = requests.get(f"{API_URL}/recommend", params=params)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logger.error(f"Error getting recommendations from API: {str(e)}")
        st.error(f"Error getting recommendations from API: {str(e)}")
        return {"recommendations": [], "query": query}

def get_recommendations_direct(query: str, max_duration: Optional[int] = None, top_k: int = 10) -> Dict[str, Any]:
    """Get recommendations directly using the recommender (no API)"""
    try:
        recommender = get_recommender()
        # Only send max_duration if positive value
        if max_duration and max_duration > 0:
            recommendations = recommender.recommend(
                query=query,
                max_duration=max_duration,
                top_k=top_k
            )
        else:
            recommendations = recommender.recommend(
                query=query,
                top_k=top_k
            )
        
        return {
            "recommendations": recommendations,
            "query": query,
            "max_duration": max_duration
        }
    except Exception as e:
        logger.error(f"Error getting recommendations directly: {str(e)}")
        st.error(f"Error getting recommendations: {str(e)}")
        return {"recommendations": [], "query": query}

def show_recommendations():
    """Input: Text/URL | Output: Sortable table with hyperlinks"""
    # Main header
    st.markdown('<div class="main-header">SHL Assessment Recommender</div>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    This tool helps you find the most relevant SHL assessments for your hiring needs. 
    Enter a job description or query about the role, and we'll recommend the best assessments.
    """)
    
    # Sidebar for input method selection
    st.sidebar.markdown("## Input Options")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Text Query", "Job Description URL"]
    )
    
    # Sidebar for advanced options
    st.sidebar.markdown("## Search Options")
    max_duration = st.sidebar.slider(
        "Max Duration (minutes)",
        min_value=0,
        max_value=120,
        value=60,
        step=5,
        help="Maximum assessment duration (0 = no limit)"
    )
    if max_duration == 0:
        max_duration = None
    
    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5
    )
    
    # API vs. Direct mode indicator
    st.sidebar.markdown("## System Status")

    # Check API availability first
    api_available = False
    api_status = st.sidebar.empty()

    try:
        response = requests.get(f"{API_URL}/model-info", timeout=3)
        if response.status_code == 200:
            model_info = response.json()
            api_status.success(f"Connected to API ({model_info.get('assessments_count', '?')} assessments)")
            api_available = True
        else:
            api_status.error(f"API available but returned error ({response.status_code})")
            api_available = False
    except Exception as e:
        logger.error(f"Cannot connect to API: {str(e)}")
        api_status.error("Cannot connect to API - Using local mode")
        api_available = False

    # Use direct mode if API is not available and direct mode is possible
    use_direct = (not api_available) or DIRECT_MODE
    if use_direct and DIRECT_MODE:
        st.sidebar.success("Running in direct mode (recommender loaded)")
    
    # Input form
    with st.container():
        if input_method == "Text Query":
            query = st.text_area(
                "Enter job description or query:",
                height=150,
                placeholder="Example: I need a cognitive assessment for software engineers that tests problem-solving and can be completed in under 30 minutes."
            )
            
            submit_button = st.button("Get Recommendations")
            
            if submit_button and query:
                with st.spinner("Generating recommendations..."):
                    if use_direct and DIRECT_MODE:
                        results = get_recommendations_direct(query, max_duration, top_k)
                    else:
                        results = get_recommendations_from_api(query, max_duration, top_k)
                    
                    display_recommendations(results)
            
        else:  # URL input
            url = st.text_input(
                "Enter job posting URL:",
                placeholder="https://www.example.com/job-posting"
            )
            
            submit_button = st.button("Extract & Get Recommendations")
            
            if submit_button and url:
                with st.spinner("Extracting text from URL..."):
                    query = extract_text_from_url(url)
                    
                if query:
                    # Show extracted text
                    with st.expander("Extracted text from URL", expanded=False):
                        st.markdown(f"<div class='url-info'>The following text was extracted from {url}:</div>", unsafe_allow_html=True)
                        st.text(query[:500] + "..." if len(query) > 500 else query)
                    
                    with st.spinner("Generating recommendations..."):
                        if use_direct and DIRECT_MODE:
                            results = get_recommendations_direct(query, max_duration, top_k)
                        else:
                            results = get_recommendations_from_api(query, max_duration, top_k)
                        
                        display_recommendations(results)
    
    # Footer
    st.markdown("""
    <div class="footer">
        SHL Assessment Recommender | Powered by AI
    </div>
    """, unsafe_allow_html=True)

def display_recommendations(results: Dict[str, Any]):
    """Display the recommendations in a visually appealing format"""
    recommendations = results.get("recommendations", [])
    
    if not recommendations:
        st.warning("No matching assessments found. Try adjusting your query or search options.")
        return
    
    st.markdown('<div class="sub-header">Recommended Assessments</div>', unsafe_allow_html=True)
    
    # Create a table for the recommendations
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Sorted by relevance")
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Relevance", "Duration", "Name"],
            index=0
        )
    
    # Sort recommendations based on selected criteria
    if sort_by == "Duration":
        sorted_recs = sorted(recommendations, key=lambda x: x.get("duration", 999) if x.get("duration") is not None else 999)
    elif sort_by == "Name":
        sorted_recs = sorted(recommendations, key=lambda x: x.get("name", ""))
    else:  # Default: Relevance
        sorted_recs = sorted(recommendations, key=lambda x: x.get("score", 0), reverse=True)
    
    # Display each recommendation
    for i, rec in enumerate(sorted_recs):
        with st.container():
            st.markdown(f'<div class="assessment-card">', unsafe_allow_html=True)
            
            # Title and score
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<div class="assessment-title">{i+1}. {rec.get("name", "Unknown")}</div>', unsafe_allow_html=True)
            with col2:
                score = rec.get("score", 0)
                st.markdown(f'<div class="assessment-score">Score: {score:.2f}</div>', unsafe_allow_html=True)
            
            # Details
            test_types = rec.get("test_type", [])
            test_types_str = ", ".join(test_types) if test_types else "N/A"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="assessment-meta">Type: {test_types_str}</div>', unsafe_allow_html=True)
            with col2:
                duration = rec.get("duration", "Unknown")
                st.markdown(f'<div class="assessment-meta">Duration: {duration} min</div>', unsafe_allow_html=True)
            with col3:
                remote = "Yes" if rec.get("remote_testing", False) else "No"
                adaptive = "Yes" if rec.get("adaptive_irt", False) else "No"
                st.markdown(f'<div class="assessment-meta">Remote: {remote} | Adaptive: {adaptive}</div>', unsafe_allow_html=True)
            
            # URL as button
            url = rec.get("url", "#")
            st.markdown(f'<a href="{url}" target="_blank"><button style="background-color: #0066b2; color: white; border: none; border-radius: 4px; padding: 0.3rem 0.8rem; cursor: pointer; font-size: 0.8rem;">View Assessment</button></a>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a download button for the results
    df = pd.DataFrame(sorted_recs)
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="shl_recommendations.csv",
        mime="text/csv",
    )

def main():
    """Main function to run the Streamlit app"""
    show_recommendations()

if __name__ == "__main__":
    main() 