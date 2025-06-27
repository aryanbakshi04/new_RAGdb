import logging
import aiohttp
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from .config import Config

logger = logging.getLogger(__name__)

class SansadClient:
    """Client for interacting with the Sansad.in API and downloading PDFs"""
    
    def __init__(self):
        self.base_url = Config.SANSAD_API_URL
        self.pdf_base_url = Config.PDF_BASE_URL
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Ensure PDF cache directory exists
        os.makedirs(Config.PDF_CACHE_DIR, exist_ok=True)
    
    def _format_pdf_url(self, pdf_url: str) -> str:
        """Format PDF URL to be accessible"""
        if not pdf_url:
            return ""
            
        # Remove localhost URL if present
        if "localhost" in pdf_url:
            pdf_url = pdf_url.split("getFile/")[-1]
            
        # If already a full URL, return as is
        if pdf_url.startswith("http"):
            return pdf_url
            
        # Construct the correct Sansad URL
        if pdf_url.startswith("/"):
            pdf_url = pdf_url[1:]
            
        # Ensure we have the full domain
        return (
            f"https://sansad.in/{pdf_url}"
            if not pdf_url.startswith("sansad.in")
            else f"https://{pdf_url}"
        )
    
    async def fetch_questions(
        self, 
        ministry: str = None, 
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """Fetch questions from the API with retries"""
        params = {
            "loksabhaNo": Config.DEFAULT_LOK_SABHA,
            "sessionNumber": Config.DEFAULT_SESSION,
            "pageNo": page,
            "pageSize": Config.DEFAULT_PAGE_SIZE,
            "locale": "en"
        }
        
        if ministry:
            params["ministry"] = ministry
            
        retry_count = 0
        max_retries = Config.MAX_RETRIES
        delay = Config.RATE_LIMIT_DELAY
        
        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.base_url,
                        params=params,
                        headers=self.headers,
                        timeout=Config.TIMEOUT
                    ) as response:
                        if response.status == 429:  # Rate limited
                            retry_count += 1
                            wait_time = delay * (2**retry_count)  # Exponential backoff
                            logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        response.raise_for_status()
                        data = await response.json()
                        
                        if not data:
                            logger.warning(f"Empty response for ministry: {ministry}")
                            return []
                            
                        logger.info(f"Successfully fetched questions for {ministry} (page {page})")
                        return await self._process_response(data)
                        
            except aiohttp.ClientResponseError as e:
                retry_count += 1
                logger.warning(f"Response error (attempt {retry_count}): {e}")
                if retry_count == max_retries:
                    logger.error(f"Failed to fetch questions after {max_retries} attempts: {e}")
                    return []
                await asyncio.sleep(delay * (2**retry_count))
                
            except aiohttp.ClientError as e:
                retry_count += 1
                logger.warning(f"Client error (attempt {retry_count}): {e}")
                if retry_count == max_retries:
                    logger.error(f"Failed to fetch questions after {max_retries} attempts: {e}")
                    return []
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Unexpected error fetching questions: {e}")
                return []
    
    async def download_pdf(self, pdf_url: str) -> Optional[str]:
        """Download and cache PDF file with error handling"""
        if not pdf_url:
            return None
            
        try:
            # Format the PDF URL properly
            formatted_url = self._format_pdf_url(pdf_url)
            
            # Extract filename from URL
            parsed_url = urlparse(formatted_url)
            filename = os.path.basename(parsed_url.path)
            
            # Ensure filename ends with .pdf
            if not filename.endswith(".pdf"):
                filename = f"{filename}.pdf"
                
            # Create file path
            file_path = Path(Config.PDF_CACHE_DIR) / filename
            
            # Check if already downloaded
            if file_path.exists():
                logger.info(f"Using cached PDF: {filename}")
                return str(file_path)
                
            logger.info(f"Downloading PDF from: {formatted_url}")
            
            # Try with retries
            retry_count = 0
            while retry_count < Config.MAX_RETRIES:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            formatted_url, 
                            headers=self.headers, 
                            timeout=Config.TIMEOUT
                        ) as response:
                            if response.status == 404:
                                logger.error(f"PDF not found: {formatted_url}")
                                return None
                                
                            if response.status == 429:  # Rate limited
                                retry_count += 1
                                wait_time = Config.RATE_LIMIT_DELAY * (2**retry_count)
                                logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
                                await asyncio.sleep(wait_time)
                                continue
                                
                            response.raise_for_status()
                            content = await response.read()
                            
                            # Basic check for PDF content
                            if not content.startswith(b"%PDF"):
                                logger.error(f"Downloaded content is not a PDF: {formatted_url}")
                                return None
                                
                            # Save the PDF
                            with open(file_path, "wb") as f:
                                f.write(content)
                                
                            logger.info(f"Successfully downloaded PDF to: {file_path}")
                            return str(file_path)
                            
                except aiohttp.ClientResponseError as e:
                    retry_count += 1
                    logger.warning(f"Response error downloading PDF (attempt {retry_count}): {e}")
                    if retry_count == Config.MAX_RETRIES:
                        logger.error(f"Failed to download PDF after {Config.MAX_RETRIES} attempts: {e}")
                        return None
                    await asyncio.sleep(Config.RATE_LIMIT_DELAY)
                    
                except aiohttp.ClientError as e:
                    retry_count += 1
                    logger.warning(f"Client error downloading PDF (attempt {retry_count}): {e}")
                    if retry_count == Config.MAX_RETRIES:
                        logger.error(f"Failed to download PDF after {Config.MAX_RETRIES} attempts: {e}")
                        return None
                    await asyncio.sleep(Config.RATE_LIMIT_DELAY)
                    
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None
    
    async def _process_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Process API response with error handling"""
        processed_questions = []
        
        try:
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                        
                    questions = item.get("listOfQuestions", [])
                    for q in questions:
                        if not isinstance(q, dict):
                            continue
                            
                        # Ensure ministry exists
                        ministry = q.get("ministry")
                        if not ministry:
                            continue
                            
                        # Clean up PDF URL
                        pdf_url = q.get("questionsFilePath", "")
                        
                        processed_q = {
                            "question_no": q.get("quesNo", ""),
                            "subject": q.get("subjects", ""),
                            "ministry": ministry,
                            "question_text": q.get("questionText", ""),
                            "pdf_url": pdf_url,
                            "date": q.get("date", ""),
                            "session": q.get("sessionNo", ""),
                        }
                        
                        processed_questions.append(processed_q)
                        
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            
        return processed_questions