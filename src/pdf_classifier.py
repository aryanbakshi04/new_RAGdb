import re
import logging
import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from typing import Tuple, Dict, List
from .config import Config

logger = logging.getLogger(__name__)

class PDFClassifier:
    """Handles the classification of PDFs by ministry based on content analysis"""
    
    def __init__(self):
        self.ministry_patterns = self._build_ministry_patterns()
        
    def _build_ministry_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build regex patterns for each ministry"""
        patterns = {}
        
        for ministry in Config.MINISTRIES:
            ministry_patterns = []
            
            # Pattern 1: Exact ministry name
            ministry_patterns.append(re.compile(r'\b' + re.escape(ministry) + r'\b', re.IGNORECASE))
            
            # Pattern 2: Without "Ministry of" prefix
            if ministry.startswith("Ministry of "):
                shortened = ministry[12:]
                ministry_patterns.append(re.compile(r'\b' + re.escape(shortened) + r'\b', re.IGNORECASE))
            
            # Pattern 3: Handle abbreviations
            if " and " in ministry.lower():
                abbr_parts = [word[0].upper() for word in ministry.split() if word.lower() not in ('of', 'the', 'and')]
                if len(abbr_parts) >= 2:
                    abbr = ''.join(abbr_parts)
                    ministry_patterns.append(re.compile(r'\b' + re.escape(abbr) + r'\b'))
            
            patterns[ministry] = ministry_patterns
            
        return patterns
    
    def classify_pdf(self, pdf_path: str) -> Tuple[str, float]:
        """
        Classify a PDF by examining its content to determine the ministry
        Returns: (ministry_name, confidence_score)
        """
        try:
            # Attempt to detect ministry from first page
            ministry, confidence = self._detect_from_first_page(pdf_path)
            
            # If confident enough, return the result
            if confidence >= 0.7:
                return ministry, confidence
            
            # If low confidence, try the alternative method
            alt_ministry, alt_confidence = self._detect_from_content_keywords(pdf_path)
            
            # Return the detection with higher confidence
            if alt_confidence > confidence:
                return alt_ministry, alt_confidence
            
            return ministry, confidence
            
        except Exception as e:
            logger.error(f"Error classifying PDF {pdf_path}: {e}")
            return "Unknown Ministry", 0.0
    
    def _detect_from_first_page(self, pdf_path: str) -> Tuple[str, float]:
        """Analyze the first page for ministry headers"""
        try:
            # Load PDF with PyPDFLoader - just the first page
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                return "Unknown Ministry", 0.0
                
            # Get first page content
            first_page = pages[0].page_content.strip()
            
            if not first_page:
                return "Unknown Ministry", 0.0
                
            # Check for common header patterns in government documents
            header_patterns = [
                # "GOVERNMENT OF INDIA\nMINISTRY OF..."
                r'(?:GOVERNMENT\s+OF\s+INDIA|भारत\s+सरकार)[\s\n]+(.*?)(?:\n|$)',
                
                # "MINISTRY OF..." at start
                r'MINISTRY\s+OF\s+(.*?)(?:\n|$)',
                
                # Department pattern
                r'DEPARTMENT\s+OF\s+(.*?)(?:\n|$)'
            ]
            
            # Check each header pattern
            for pattern in header_patterns:
                header_match = re.search(pattern, first_page, re.IGNORECASE)
                if header_match:
                    header_text = header_match.group(1).strip()
                    
                    # Check if header contains any ministry name
                    for ministry, patterns in self.ministry_patterns.items():
                        for pattern in patterns:
                            if pattern.search(header_text):
                                return ministry, 0.9  # High confidence for header match
                    
                    # If not found but we have header text, try text similarity
                    best_match = self._find_best_ministry_match(header_text)
                    if best_match[1] > 0.5:  # Reasonable similarity threshold
                        return best_match
            
            # If no header match, check for ministry mentions in first 500 chars
            preview = first_page[:500].lower()
            for ministry, patterns in self.ministry_patterns.items():
                for pattern in patterns:
                    if pattern.search(preview):
                        # Early mention gets higher confidence
                        position = preview.find(ministry.lower())
                        if position >= 0:
                            # Earlier mentions get higher confidence
                            confidence = 0.8 - (position / 1000)
                            return ministry, max(0.5, confidence)
            
            return "Unknown Ministry", 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing first page of {pdf_path}: {e}")
            return "Unknown Ministry", 0.0
    
    def _detect_from_content_keywords(self, pdf_path: str) -> Tuple[str, float]:
        """Analyze document content for ministry-related keywords"""
        try:
            # Load PDF with PyPDFLoader - all pages
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                return "Unknown Ministry", 0.0
            
            # Get content from first few pages (up to 3)
            content = ""
            for page in pages[:min(3, len(pages))]:
                content += page.page_content + "\n"
                
            content = content.lower()
            
            # Score each ministry based on keyword frequency
            ministry_scores = {}
            
            for ministry in Config.MINISTRIES:
                # Get keywords specific to this ministry
                keywords = self._extract_ministry_keywords(ministry)
                score = 0
                
                for keyword in keywords:
                    occurrences = content.count(keyword.lower())
                    # Weight by keyword specificity (length)
                    score += occurrences * (len(keyword) / 5)
                
                if score > 0:
                    ministry_scores[ministry] = score
            
            # Find ministry with highest score
            if ministry_scores:
                top_ministry = max(ministry_scores.items(), key=lambda x: x[1])
                
                # Normalize score to confidence (0-1)
                confidence = min(0.8, top_ministry[1] / 10)  # Cap at 0.8
                
                return top_ministry[0], confidence
            
            return "Unknown Ministry", 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing content of {pdf_path}: {e}")
            return "Unknown Ministry", 0.0
    
    def _extract_ministry_keywords(self, ministry: str) -> List[str]:
        """Extract keywords from ministry name"""
        keywords = []
        
        # Remove "Ministry of" if present
        name = ministry.lower()
        if name.startswith("ministry of "):
            name = name[12:]
        
        # Split into words
        words = name.split()
        
        # Add individual words (except common ones)
        common_words = {"of", "and", "the", "in", "for"}
        for word in words:
            if word not in common_words and len(word) > 3:
                keywords.append(word)
        
        # Add the whole name
        keywords.append(name)
        
        # Add original ministry name
        keywords.append(ministry.lower())
        
        return keywords
    
    def _find_best_ministry_match(self, text: str) -> Tuple[str, float]:
        """Find best matching ministry using text similarity"""
        best_ministry = "Unknown Ministry"
        best_score = 0
        
        text = text.lower()
        
        for ministry in Config.MINISTRIES:
            # For ministries, compare with the part after "Ministry of"
            if ministry.startswith("Ministry of "):
                compare_text = ministry[12:].lower()
            else:
                compare_text = ministry.lower()
                
            # Calculate similarity
            similarity = self._text_similarity(compare_text, text)
            
            if similarity > best_score:
                best_score = similarity
                best_ministry = ministry
        
        return best_ministry, best_score
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple word overlap approach
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common words
        common_words = {'of', 'the', 'and', 'in', 'for', 'to', 'with'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate overlap
        overlap = len(words1.intersection(words2))
        return overlap / max(len(words1), len(words2))
    
    def organize_pdf(self, pdf_path: str, target_dir: Path = None) -> Dict:
        """
        Organize a PDF into the appropriate ministry directory
        Returns metadata about the classification
        """
        try:
            # Classify the PDF
            ministry, confidence = self.classify_pdf(pdf_path)
            
            # Use default target directory if not provided
            if target_dir is None:
                target_dir = Config.get_ministry_dir(ministry)
                
            # Create target directory if it doesn't exist
            target_dir.mkdir(exist_ok=True, parents=True)
            
            # Get PDF filename
            pdf_name = os.path.basename(pdf_path)
            
            # Create target path
            target_path = target_dir / pdf_name
            
            # Copy the PDF
            import shutil
            shutil.copy2(pdf_path, target_path)
            
            # Create metadata
            metadata = {
                "ministry": ministry,
                "confidence": confidence,
                "original_path": str(pdf_path),
                "classified_by": Config.CURRENT_USER,
                "classified_at": Config.CURRENT_TIME,
                "classification_method": "content_analysis"
            }
            
            # Save metadata file
            metadata_path = target_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Organized PDF {pdf_name} into {ministry} with confidence {confidence:.2f}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error organizing PDF {pdf_path}: {e}")
            return {"error": str(e), "ministry": "Unknown Ministry", "confidence": 0.0}