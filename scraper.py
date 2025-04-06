import json
import logging
import os
import re
import asyncio
import random
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urljoin
import httpx
from selectolax.parser import HTMLParser
from playwright.async_api import async_playwright
import tqdm
import pandas as pd
from pydantic import BaseModel, Field
from datetime import datetime

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
    MAX_RETRIES = 5
    MIN_RETRY_DELAY = 2  # seconds
    MAX_RETRY_DELAY = 10  # seconds
    
    def __init__(self, use_playwright: bool = True, output_dir: str = "./", max_depth: int = 5):
        """
        Initialize the SHL Scraper
        
        Args:
            use_playwright: Whether to use Playwright for JS rendering
            output_dir: Directory to save output files
            max_depth: Maximum depth for pagination (default: 5)
        """
        self.use_playwright = use_playwright
        self.output_dir = output_dir
        self.max_depth = max_depth
        self.assessments: List[Assessment] = []
        self.processed_urls: Set[str] = set()
        self.httpx_client = httpx.AsyncClient(
            follow_redirects=True, 
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        self.browser = None
        self.page = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    async def __aenter__(self):
        if self.use_playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.page = await self.browser.new_page()
            await self.page.set_viewport_size({"width": 1280, "height": 800})
            
            # Set user agent
            await self.page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            })
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.httpx_client.aclose()
        if self.browser:
            await self.browser.close()
            await self.playwright.stop()
    
    async def fetch_with_retry(self, url: str) -> str:
        """Fetch a URL with exponential backoff retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.use_playwright and self.page:
                    logger.info(f"Fetching {url} with Playwright (attempt {attempt + 1})")
                    await self.page.goto(url, wait_until="networkidle")
                    # Add random delay to avoid being blocked
                    await asyncio.sleep(random.uniform(1, 3))
                    content = await self.page.content()
                    return content
                else:
                    logger.info(f"Fetching {url} with httpx (attempt {attempt + 1})")
                    response = await self.httpx_client.get(url)
                    response.raise_for_status()
                    return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed for {url}: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(self.MAX_RETRY_DELAY, 
                               self.MIN_RETRY_DELAY * (2 ** attempt)) * (0.5 + random.random())
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {url} after {self.MAX_RETRIES} attempts")
                    raise
    
    async def detect_pagination(self, html: str) -> List[str]:
        """
        Detect pagination URLs from HTML
        
        Args:
            html: HTML content
            
        Returns:
            List of pagination URLs
        """
        parser = HTMLParser(html)
        pagination_urls = []
        
        # Look for pagination links
        pagination_elements = parser.css("nav.pagination, .pagination, .pager, .paging")
        for element in pagination_elements:
            # Look for links with numbers or next/previous
            links = element.css("a")
            for link in links:
                href = link.attributes.get("href", "")
                text = link.text().strip().lower()
                
                # Skip if no href or if it's the current page
                if not href or "current" in text or "active" in text:
                    continue
                
                # Skip if it's a JavaScript link
                if href.startswith("javascript:"):
                    continue
                
                # Make URL absolute if needed
                if href and not href.startswith("http"):
                    href = urljoin(self.BASE_URL, href)
                
                # Add to list if not already processed
                if href and href not in self.processed_urls:
                    pagination_urls.append(href)
        
        # Also look for "Load More" or "Show More" buttons
        load_more_buttons = parser.css("button.load-more, .load-more, .show-more")
        for button in load_more_buttons:
            data_url = button.attributes.get("data-url", "")
            if data_url and not data_url.startswith("http"):
                data_url = urljoin(self.BASE_URL, data_url)
            if data_url and data_url not in self.processed_urls:
                pagination_urls.append(data_url)
        
        return pagination_urls
    
    def parse_assessment_table(self, html: str) -> List[Dict[str, Any]]:
        """
        Parse assessment tables from HTML
        
        Args:
            html: HTML content
            
        Returns:
            List of assessment dictionaries
        """
        parser = HTMLParser(html)
        tables = parser.css("table.products-table, table.product-catalog, .product-grid, .assessment-grid")
        
        if not tables:
            logger.warning("No specific product tables found in HTML")
            # Look for tables with certain column patterns that match product tables
            all_tables = parser.css("table")
            
            for table in all_tables:
                headers = table.css("th")
                header_texts = [h.text().strip().lower() for h in headers]
                header_text = " ".join(header_texts)
                
                # Check if this looks like a product table based on headers
                if any(x in header_text for x in ["product name", "test name", "assessment name", "name"]) and \
                   any(x in header_text for x in ["remote", "online", "testing", "type", "duration"]):
                    tables = [table]
                    logger.info(f"Found product table with headers: {header_texts}")
                    break
            
            if not tables:
                # Look for product cards/grids
                product_cards = parser.css(".product-card, .assessment-card, .test-card")
                if product_cards:
                    tables = [product_cards]
                    logger.info("Found product cards grid")
                else:
                    tables = all_tables  # Fallback to any table as last resort
        
        all_assessments = []
        
        for table in tables:
            # Check if this is a product grid/cards view
            if table.tag == "div" and any(cls in table.attributes.get("class", "").lower() 
                                        for cls in ["grid", "cards", "products"]):
                # Parse product cards
                cards = table.css(".product-card, .assessment-card, .test-card")
                for card in cards:
                    assessment = self._parse_product_card(card)
                    if assessment:
                        all_assessments.append(assessment)
                continue
            
            # Parse table view
            headers = table.css("th")
            header_texts = [h.text().strip().lower() for h in headers]
            
            # Skip if this doesn't look like the assessment table
            if not any(text in " ".join(header_texts) for text in ["name", "product", "remote", "test type"]):
                logger.debug(f"Skipping table with headers: {header_texts}")
                continue
                
            rows = table.css("tr")
            # Skip header row
            for row in rows[1:]:
                cells = row.css("td")
                if not cells or len(cells) < 4:
                    continue
                
                assessment = self._parse_table_row(cells)
                if assessment:
                    all_assessments.append(assessment)
        
        return all_assessments

    def _parse_product_card(self, card) -> Optional[Dict[str, Any]]:
        """Parse a product card element"""
        try:
            # Get name and URL
            name_elem = card.css_first("h2, h3, .product-title, .assessment-title")
            name = name_elem.text().strip() if name_elem else ""
            
            link = card.css_first("a")
            url = link.attributes.get("href", "") if link else ""
            if url and not url.startswith("http"):
                url = urljoin(self.BASE_URL, url)
            
            # Get test type
            type_elem = card.css_first(".test-type, .assessment-type, .product-type")
            test_types = []
            if type_elem:
                type_text = type_elem.text().strip()
                test_types = [t.strip() for t in type_text.split(",")]
            
            # Get duration
            duration_elem = card.css_first(".duration, .time, .length")
            duration = None
            if duration_elem:
                duration_text = duration_elem.text().strip()
                matches = re.search(r'(\d+)\s*min', duration_text, re.IGNORECASE)
                if matches:
                    duration = int(matches.group(1))
            
            # Check for remote testing and adaptive features
            remote_testing = bool(card.css_first(".remote, .online, .virtual"))
            adaptive_irt = bool(card.css_first(".adaptive, .irt, .smart"))
            
            if name and url:
                return {
                    "name": name,
                    "url": url,
                    "remote_testing": remote_testing,
                    "adaptive_irt": adaptive_irt,
                    "duration": duration,
                    "test_type": test_types
                }
        except Exception as e:
            logger.error(f"Error parsing product card: {str(e)}")
        return None

    def _parse_table_row(self, cells) -> Optional[Dict[str, Any]]:
        """Parse a table row into an assessment dictionary"""
        try:
            # Extract assessment name and URL
            name_cell = cells[0]
            name_link = name_cell.css_first("a")
            
            if not name_link:
                return None
            
            name = name_link.text().strip()
            url = name_link.attributes.get("href", "")
            
            if url and not url.startswith("http"):
                url = urljoin(self.BASE_URL, url)
            
            # Extract remote testing
            remote_cell = cells[1] if len(cells) > 1 else None
            remote_testing = False
            if remote_cell:
                remote_img = remote_cell.css_first("img")
                remote_icon = remote_cell.css_first("i.fa-check, .check-icon")
                remote_testing = bool(remote_img or remote_icon or "✓" in remote_cell.text())
            
            # Extract adaptive/IRT
            adaptive_cell = cells[2] if len(cells) > 2 else None
            adaptive_irt = False
            if adaptive_cell:
                adaptive_img = adaptive_cell.css_first("img")
                adaptive_icon = adaptive_cell.css_first("i.fa-check, .check-icon")
                adaptive_irt = bool(adaptive_img or adaptive_icon or "✓" in adaptive_cell.text())
            
            # Parse test type
            test_type_cell = cells[3] if len(cells) > 3 else None
            test_types = []
            
            if test_type_cell:
                test_type_text = test_type_cell.text().strip()
                
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
                
                # First try to interpret as codes
                has_matched_code = False
                for code in test_type_text:
                    if code in type_map:
                        test_types.append(type_map[code])
                        has_matched_code = True
                
                # If no codes matched, use the full text
                if not has_matched_code and test_type_text:
                    # Check if there are multiple types separated by commas
                    if ',' in test_type_text:
                        test_types = [t.strip() for t in test_type_text.split(',')]
                    else:
                        test_types = [test_type_text]
                
                # Special case for technical skills
                for tech_skill in ["Java", "Python", "SQL", "JavaScript", ".NET", "C#"]:
                    if tech_skill.lower() in name.lower():
                        if "Knowledge & Skills" not in test_types:
                            test_types.append("Knowledge & Skills")
                        if "Technical Skills" not in test_types:
                            test_types.append("Technical Skills")
            
            # Ensure we have at least one test type
            if not test_types:
                test_types = ["Unspecified"]
            
            return {
                "name": name,
                "url": url,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "test_type": test_types,
                "duration": None  # Will be extracted separately
            }
        except Exception as e:
            logger.error(f"Error parsing table row: {str(e)}")
            return None
    
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
            
            # First try to find duration in specific elements
            duration_elements = parser.css(".duration, .assessment-duration, .time-duration")
            for element in duration_elements:
                text = element.text()
                matches = re.search(r'(\d+)\s*min', text, re.IGNORECASE)
                if matches:
                    return int(matches.group(1))
            
            # If not found in specific elements, try full text search
            text = parser.text()
            
            # Try to find duration information
            duration_patterns = [
                r'(\d+)\s*minutes',
                r'(\d+)\s*min',
                r'Duration:?\s*(\d+)',
                r'Time:?\s*(\d+)',
                r'approximately (\d+)',
                r'takes (\d+) min',
                r'completed in (\d+)',
                r'length: (\d+)'
            ]
            
            for pattern in duration_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        # Get the first number
                        return int(matches[0])
                    except (ValueError, IndexError):
                        pass
            
            # If still not found, look for duration in page description
            meta_description = parser.css_first('meta[name="description"]')
            if meta_description:
                description = meta_description.attributes.get("content", "")
                for pattern in duration_patterns:
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    if matches:
                        try:
                            return int(matches[0])
                        except (ValueError, IndexError):
                            pass
            
            # If we still don't have a duration, make educated guesses based on assessment type
            name = parser.css_first('h1, .product-title')
            if name:
                name_text = name.text().lower()
                # Set default durations based on assessment type
                if any(term in name_text for term in ['personality', 'questionnaire', 'behavior']):
                    return 25  # Personality assessments typically take 25 minutes
                elif any(term in name_text for term in ['ability', 'aptitude', 'reasoning']):
                    return 30  # Ability tests typically take 30 minutes
                elif any(term in name_text for term in ['situational', 'judgment']):
                    return 20  # SJTs typically take 20 minutes
                
            return None
        except Exception as e:
            logger.error(f"Error extracting duration for {url}: {str(e)}")
            return None
    
    async def scrape_page(self, url: str, current_depth: int = 0) -> List[Dict[str, Any]]:
        """
        Scrape a single page
        
        Args:
            url: URL to scrape
            current_depth: Current depth level (used for recursion limits)
            
        Returns:
            List of assessment dictionaries
        """
        logger.info(f"Scraping page: {url} (depth: {current_depth}/{self.max_depth})")
        html = await self.fetch_with_retry(url)
        
        # Parse assessment table
        assessments = self.parse_assessment_table(html)
        logger.info(f"Found {len(assessments)} assessments on {url}")
        
        # Don't go deeper if we're at max depth
        all_assessments = assessments.copy()
        if current_depth < self.max_depth:
            # Detect pagination
            pagination_urls = await self.detect_pagination(html)
            logger.info(f"Found {len(pagination_urls)} pagination URLs")
            
            # Process pagination
            for pagination_url in pagination_urls:
                if pagination_url not in self.processed_urls:
                    self.processed_urls.add(pagination_url)
                    try:
                        page_assessments = await self.scrape_page(pagination_url, current_depth + 1)
                        all_assessments.extend(page_assessments)
                    except Exception as e:
                        logger.error(f"Error scraping pagination URL {pagination_url}: {str(e)}")
        
        return all_assessments
    
    async def scrape_all_shl_assessments(self) -> List[Assessment]:
        """
        Scrape all SHL assessments
        
        Returns:
            List of Assessment objects
        """
        logger.info(f"Starting to scrape SHL assessments (max depth: {self.max_depth})")
        self.processed_urls = set()
        
        # Start with the main catalog page
        self.processed_urls.add(self.BASE_URL)
        assessments_dicts = await self.scrape_page(self.BASE_URL)
        
        # Extract unique assessments by URL
        unique_assessments = {}
        for assessment in assessments_dicts:
            if assessment["url"] not in unique_assessments:
                unique_assessments[assessment["url"]] = assessment
        
        assessments_dicts = list(unique_assessments.values())
        logger.info(f"Found {len(assessments_dicts)} unique assessments")
        
        # Extract duration for each assessment
        logger.info("Extracting durations...")
        durations = []
        for i, a in enumerate(assessments_dicts):
            logger.info(f"Extracting duration for assessment {i+1}/{len(assessments_dicts)}: {a['name']}")
            duration = await self.extract_duration(a["url"])
            durations.append(duration)
        
        # Assign durations to assessments
        for i, duration in enumerate(durations):
            assessments_dicts[i]["duration"] = duration
        
        # Convert to Assessment objects
        self.assessments = [Assessment(**a) for a in assessments_dicts]
        
        # Log summary of data quality
        with_duration = sum(1 for a in self.assessments if a.duration is not None)
        remote_count = sum(1 for a in self.assessments if a.remote_testing)
        adaptive_count = sum(1 for a in self.assessments if a.adaptive_irt)
        logger.info(f"Data quality: {with_duration}/{len(self.assessments)} have duration, "
                   f"{remote_count} remote testing, {adaptive_count} adaptive/IRT")
        
        return self.assessments
    
    def save_to_json(self, filename: str = None) -> str:
        """
        Save assessments to JSON file
        
        Args:
            filename: Optional filename, defaults to timestamped file
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"shl_assessments_{timestamp}.json")
        else:
            filename = os.path.join(self.output_dir, filename)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([a.model_dump() for a in self.assessments], f, indent=2)
        
        logger.info(f"Saved {len(self.assessments)} assessments to {filename}")
        
        # Also save to a standard filename for application use
        standard_filename = os.path.join(self.output_dir, "shl_assessments.json")
        with open(standard_filename, "w", encoding="utf-8") as f:
            json.dump([a.model_dump() for a in self.assessments], f, indent=2)
        
        return filename
    
    def save_to_csv(self, filename: str = None) -> str:
        """
        Save assessments to CSV file
        
        Args:
            filename: Optional filename, defaults to timestamped file
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"shl_assessments_{timestamp}.csv")
        else:
            filename = os.path.join(self.output_dir, filename)
        
        # Convert to DataFrame
        assessments_dicts = []
        for a in self.assessments:
            a_dict = a.model_dump()
            a_dict["test_type"] = ", ".join(a_dict["test_type"])
            assessments_dicts.append(a_dict)
        
        df = pd.DataFrame(assessments_dicts)
        df.to_csv(filename, index=False)
        
        logger.info(f"Saved {len(self.assessments)} assessments to {filename}")
        
        # Also save to a standard filename for application use
        standard_filename = os.path.join(self.output_dir, "shl_assessments.csv")
        df.to_csv(standard_filename, index=False)
        
        return filename

async def load_sample_data() -> List[Assessment]:
    """Load sample assessment data as fallback"""
    logger.info("Loading sample assessment data")
    
    sample_assessments = [
        {
            "name": "Verify Numerical Reasoning Test",
            "url": "https://www.shl.com/solutions/products/verify-numerical-reasoning-test/",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 25,
            "test_type": ["Ability & Aptitude", "Numerical Reasoning"]
        },
        {
            "name": "Verify Verbal Reasoning Test",
            "url": "https://www.shl.com/solutions/products/verify-verbal-reasoning-test/",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 25,
            "test_type": ["Ability & Aptitude", "Verbal Reasoning"]
        },
        {
            "name": "Verify General Ability Test",
            "url": "https://www.shl.com/solutions/products/verify-general-ability-test/",
            "remote_testing": True,
            "adaptive_irt": False,
            "duration": 36,
            "test_type": ["Ability & Aptitude", "Numerical Reasoning", "Verbal Reasoning", "Inductive Reasoning"]
        },
        {
            "name": "Work Strengths Questionnaire",
            "url": "https://www.shl.com/solutions/products/work-strengths-questionnaire/",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": 25,
            "test_type": ["Personality & Behavior"]
        },
        {
            "name": "ADEPT-15 Personality Assessment",
            "url": "https://www.shl.com/solutions/products/adept-15-personality-assessment/",
            "remote_testing": True,
            "adaptive_irt": True,
            "duration": 25,
            "test_type": ["Personality & Behavior"]
        }
    ]
    
    return [Assessment(**assessment) for assessment in sample_assessments]

async def scrape_all_shl_assessments(force_refresh: bool = False) -> Tuple[List[Assessment], bool]:
    """
    Scrape SHL's assessment catalog or load from cached data
    
    Args:
        force_refresh: Force a fresh scrape even if cached data exists
        
    Returns:
        Tuple of (assessments list, whether data was freshly scraped)
    """
    # Check for cached data
    json_path = "shl_assessments.json"
    freshly_scraped = False
    
    if not force_refresh and os.path.exists(json_path):
        # Use cached data
        file_age_days = (datetime.now().timestamp() - os.path.getmtime(json_path)) / (60 * 60 * 24)
        logger.info(f"Found cached data ({json_path}) from {file_age_days:.1f} days ago")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                assessment_dicts = json.load(f)
            
            assessments = [Assessment(**a) for a in assessment_dicts]
            logger.info(f"Loaded {len(assessments)} assessments from cache")
            
            # Only use cache if it has reasonable number of assessments and not too old
            if len(assessments) > 5 and file_age_days < 7:
                return assessments, freshly_scraped
            else:
                logger.info(f"Cache contains only {len(assessments)} assessments or is {file_age_days:.1f} days old, refreshing...")
        except Exception as e:
            logger.error(f"Error loading cached data: {str(e)}. Will scrape fresh data.")
    
    # If we got here, we need to scrape fresh data
    try:
        logger.info("Starting fresh scrape of SHL assessments")
        async with SHLScraper(use_playwright=True) as scraper:
            assessments = await scraper.scrape_all_shl_assessments()
            
            if assessments:
                # Save the data
                scraper.save_to_json("shl_assessments.json")
                scraper.save_to_csv("shl_assessments.csv")
                freshly_scraped = True
                return assessments, freshly_scraped
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
    
    # Fallback to sample data if scraping fails
    assessments = await load_sample_data()
    return assessments, freshly_scraped

async def main():
    """Main function to run the scraper"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="SHL Assessment Scraper")
    parser.add_argument("--force", action="store_true", help="Force a fresh scrape even if cached data exists")
    parser.add_argument("--output-dir", type=str, default="./", help="Directory to save output files")
    args = parser.parse_args()
    
    # Run the scraper
    assessments, freshly_scraped = await scrape_all_shl_assessments(force_refresh=args.force)
    
    # Print summary
    print(f"{'Scraped' if freshly_scraped else 'Loaded'} {len(assessments)} assessments:")
    
    # Analyze data quality
    with_duration = sum(1 for a in assessments if a.duration is not None)
    remote_count = sum(1 for a in assessments if a.remote_testing)
    adaptive_count = sum(1 for a in assessments if a.adaptive_irt)
    
    print(f"Data quality:")
    print(f"- {with_duration}/{len(assessments)} ({with_duration/len(assessments)*100:.1f}%) have duration")
    print(f"- {remote_count}/{len(assessments)} ({remote_count/len(assessments)*100:.1f}%) support remote testing")
    print(f"- {adaptive_count}/{len(assessments)} ({adaptive_count/len(assessments)*100:.1f}%) are adaptive/IRT")
    
    # Print sample assessments
    print("\nSample assessments:")
    for i, assessment in enumerate(assessments[:5]):
        print(f"{i+1}. {assessment.name}")
        print(f"   URL: {assessment.url}")
        print(f"   Duration: {assessment.duration} minutes")
        print(f"   Remote: {assessment.remote_testing}, Adaptive: {assessment.adaptive_irt}")
        print(f"   Test Type: {', '.join(assessment.test_type)}")

if __name__ == "__main__":
    asyncio.run(main()) 