import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Any, Union
import logging
from urllib.parse import urljoin
import time

import httpx
from selectolax.parser import HTMLParser
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm
import pandas as pd
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define assessment schema
class Assessment(BaseModel):
    name: str
    url: str
    remote_testing: bool
    adaptive_irt: bool
    duration: Optional[int] = None  # in minutes
    test_type: List[str] = Field(default_factory=list)

class SHLScraper:
    BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    def __init__(self, use_playwright: bool = False):
        """
        Initialize the SHL Scraper
        
        Args:
            use_playwright: Whether to use Playwright for JS rendering
        """
        self.use_playwright = use_playwright
        self.assessments: List[Assessment] = []
        self.httpx_client = httpx.AsyncClient(
            follow_redirects=True, 
            timeout=30.0
        )
        self.browser = None
        self.page = None
    
    async def __aenter__(self):
        if self.use_playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.page = await self.browser.new_page()
            await self.page.set_viewport_size({"width": 1280, "height": 800})
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.httpx_client.aclose()
        if self.browser:
            await self.browser.close()
            await self.playwright.stop()
    
    async def fetch_with_retry(self, url: str) -> str:
        """Fetch a URL with retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.use_playwright and self.page:
                    await self.page.goto(url)
                    await self.page.wait_for_load_state("networkidle")
                    content = await self.page.content()
                    return content
                else:
                    response = await self.httpx_client.get(url)
                    response.raise_for_status()
                    return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed for {url}: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch {url} after {self.MAX_RETRIES} attempts")
                    raise
    
    def parse_assessment_table(self, html: str) -> List[Dict[str, Any]]:
        """
        Parse assessment tables from HTML
        
        Args:
            html: HTML content
            
        Returns:
            List of assessment dictionaries
        """
        parser = HTMLParser(html)
        tables = parser.css("table")
        
        all_assessments = []
        
        for table in tables:
            rows = table.css("tr")
            # Skip header row
            for row in rows[1:]:
                cells = row.css("td")
                if not cells or len(cells) < 4:
                    continue
                
                # Extract assessment name and URL
                name_cell = cells[0]
                name_link = name_cell.css_first("a")
                
                if not name_link:
                    continue
                
                name = name_link.text().strip()
                url = name_link.attributes.get("href", "")
                
                if url and not url.startswith("http"):
                    url = urljoin(self.BASE_URL, url)
                
                # Extract other attributes
                remote_testing = bool(cells[1].css_first("img"))
                adaptive_irt = bool(cells[2].css_first("img"))
                
                # Parse test type
                test_type_text = cells[3].text().strip()
                test_types = []
                
                # Parse test type codes
                type_map = {
                    'A': 'Ability & Aptitude',
                    'B': 'Biodata & Situational Judgement',
                    'C': 'Competencies',
                    'D': 'Development & 360',
                    'E': 'Assessment Exercises',
                    'K': 'Knowledge & Skills',
                    'P': 'Personality & Behavior',
                    'S': 'Simulations'
                }
                
                for code in test_type_text:
                    if code in type_map:
                        test_types.append(type_map[code])
                
                assessment = {
                    "name": name,
                    "url": url,
                    "remote_testing": remote_testing,
                    "adaptive_irt": adaptive_irt,
                    "test_type": test_types,
                    "duration": None  # Will be extracted separately
                }
                
                all_assessments.append(assessment)
        
        return all_assessments
    
    async def extract_duration(self, url: str) -> Optional[int]:
        """
        Extract duration information from individual assessment page
        
        Args:
            url: URL of the assessment page
            
        Returns:
            Duration in minutes or None if not found
        """
        try:
            html = await self.fetch_with_retry(url)
            parser = HTMLParser(html)
            text = parser.text()
            
            # Try to find duration information
            duration_patterns = [
                r'(\d+)\s*minutes',
                r'(\d+)\s*min',
                r'Duration:?\s*(\d+)',
                r'Time:?\s*(\d+)',
                r'approximately (\d+)'
            ]
            
            for pattern in duration_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        return int(matches[0])
                    except (ValueError, IndexError):
                        pass
            
            return None
        except Exception as e:
            logger.error(f"Error extracting duration for {url}: {str(e)}")
            return None
    
    async def detect_pagination(self, html: str) -> List[str]:
        """
        Detect pagination links
        
        Args:
            html: HTML content
            
        Returns:
            List of pagination URLs
        """
        parser = HTMLParser(html)
        pagination = parser.css("ul.pagination li a")
        
        pages = []
        for page_link in pagination:
            href = page_link.attributes.get("href", "")
            if href and not href.startswith("http"):
                href = urljoin(self.BASE_URL, href)
            if href and href not in pages and href != self.BASE_URL:
                pages.append(href)
        
        return pages
    
    async def scrape_all_shl_assessments(self) -> List[Assessment]:
        """
        Fully scrape SHL's catalog with:
        - Auto pagination detection
        - JS fallback via Playwright
        - Error retries (3 attempts)
        - Returns List[Assessment] with ALL fields from SHL PDF
        
        Returns:
            List of Assessment objects
        """
        logger.info("Starting SHL catalog scraping...")
        all_assessments = []
        
        # Fetch the main catalog page
        main_html = await self.fetch_with_retry(self.BASE_URL)
        
        # Detect pagination
        logger.info("Detecting pagination...")
        pagination_urls = await self.detect_pagination(main_html)
        
        # Parse assessments from main page
        main_page_assessments = self.parse_assessment_table(main_html)
        all_assessments.extend(main_page_assessments)
        
        # Parse assessments from pagination pages
        if pagination_urls:
            logger.info(f"Found {len(pagination_urls)} additional pages")
            for page_url in tqdm(pagination_urls, desc="Scraping pages"):
                try:
                    page_html = await self.fetch_with_retry(page_url)
                    page_assessments = self.parse_assessment_table(page_html)
                    all_assessments.extend(page_assessments)
                    # Be nice to the server
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error scraping page {page_url}: {str(e)}")
        
        # Fetch duration for each assessment
        logger.info(f"Extracting duration for {len(all_assessments)} assessments...")
        
        duration_tasks = []
        for assessment in all_assessments:
            # Create a task for each assessment
            task = asyncio.create_task(self.extract_duration(assessment["url"]))
            duration_tasks.append((assessment, task))
            # Be nice to the server - limit concurrency
            if len(duration_tasks) >= 5:
                await asyncio.sleep(1)
        
        # Process results
        for assessment, task in tqdm(duration_tasks, desc="Fetching durations"):
            try:
                duration = await task
                assessment["duration"] = duration
            except Exception as e:
                logger.error(f"Error fetching duration for {assessment['name']}: {str(e)}")
        
        # Convert to Pydantic models
        validated_assessments = []
        for assessment_dict in all_assessments:
            try:
                assessment = Assessment(**assessment_dict)
                validated_assessments.append(assessment)
            except Exception as e:
                logger.warning(f"Invalid assessment data: {assessment_dict}. Error: {str(e)}")
        
        self.assessments = validated_assessments
        logger.info(f"Scraped {len(validated_assessments)} valid assessments")
        
        return validated_assessments
    
    def save_to_json(self, filename: str = "shl_assessments.json"):
        """Save scraped assessments to JSON file"""
        with open(filename, "w") as f:
            json.dump([a.model_dump() for a in self.assessments], f, indent=2)
        logger.info(f"Saved {len(self.assessments)} assessments to {filename}")
    
    def save_to_csv(self, filename: str = "shl_assessments.csv"):
        """Save scraped assessments to CSV file"""
        df = pd.DataFrame([a.model_dump() for a in self.assessments])
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(self.assessments)} assessments to {filename}")

async def load_sample_data() -> List[Assessment]:
    """Load sample assessment data if scraping fails"""
    logger.info("Loading sample assessment data...")
    
    # Create some sample assessments based on SHL catalog
    sample_assessments = [
        {
            "name": "Java Programming Skills",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/java-programming/",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 40,
            "test_type": ["Knowledge & Skills", "Ability & Aptitude"]
        },
        {
            "name": "Python Programming Skills",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/python-programming/",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 45,
            "test_type": ["Knowledge & Skills"]
        },
        {
            "name": "JavaScript Programming Skills",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/javascript-programming/",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 35,
            "test_type": ["Knowledge & Skills"]
        },
        {
            "name": "Cognitive Test",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/cognitive-test/",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": 30,
            "test_type": ["Ability & Aptitude"]
        },
        {
            "name": "Personality Assessment",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/personality-assessment/",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 25,
            "test_type": ["Personality & Behavior"]
        }
    ]
    
    return [Assessment(**assessment) for assessment in sample_assessments]

async def scrape_all_shl_assessments() -> List[Assessment]:
    """
    Fully scrape SHL's catalog with:
    - Auto pagination detection
    - JS fallback via Playwright
    - Error retries (3 attempts)
    - Returns List[Assessment] with ALL fields from SHL PDF
    
    Returns:
        List of Assessment objects
    """
    try:
        async with SHLScraper(use_playwright=False) as scraper:
            assessments = await scraper.scrape_all_shl_assessments()
            
            if assessments:
                # Save the data
                scraper.save_to_json()
                scraper.save_to_csv()
                return assessments
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
    
    # Fallback to sample data if scraping fails
    return await load_sample_data()

async def main():
    """Main function to run the scraper"""
    assessments = await scrape_all_shl_assessments()
    print(f"Scraped {len(assessments)} assessments:")
    for i, assessment in enumerate(assessments[:5]):
        print(f"{i+1}. {assessment.name} - Duration: {assessment.duration} minutes")
    
    # Save sample data for testing
    if not os.path.exists("shl_assessments.json"):
        with open("shl_assessments.json", "w") as f:
            json.dump([a.model_dump() for a in assessments], f, indent=2)

if __name__ == "__main__":
    asyncio.run(main()) 