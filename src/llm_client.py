import logging
import google.generativeai as genai
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .config import Config

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with Google's Gemini model"""
    
    def __init__(self):
        try:
            # Configure API key
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            # Initialize model
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Create thread pool executor for async operations
            self.executor = ThreadPoolExecutor(max_workers=1)
            
            logger.info("Successfully initialized LLM client")
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
            raise
    
    # NEW METHOD: Synchronous wrapper for generate_response
    def generate_response_sync(self, question: str, context: List[Dict[str, Any]], ministry: str) -> str:
        """Synchronous wrapper for generate_response"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.generate_response(question, context, ministry))
            loop.close()
            return response
        except Exception as e:
            logger.error(f"Error in synchronous response generation: {e}")
            return (
                "I apologize, but I encountered an error while generating the response. "
                "Please try again with a simpler question or wait a moment before retrying."
            )
    
    async def generate_response(
        self, 
        question: str, 
        context: List[Dict[str, Any]], 
        ministry: str
    ) -> str:
        """
        Generate a response to a question using context documents
        Handles ministry-specific context to ensure relevant responses
        """
        try:
            # Construct prompt
            prompt = self._construct_prompt(question, context, ministry)
            
            # Use thread pool to run blocking API call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                self._generate_content,
                prompt
            )
            
            if not response or not response.text:
                logger.warning("Empty response from LLM")
                return (
                    "I apologize, but I couldn't generate a meaningful response. "
                    "Please try rephrasing your question."
                )
            
            # Format response
            formatted_response = self._format_response(response.text, context)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return (
                "I apologize, but I encountered an error while generating the response. "
                "This might be due to connection issues or service limitations. "
                "Please try again with a simpler question or wait a moment before retrying."
            )
    
    def _generate_content(self, prompt: str):
        """Send prompt to Gemini model and get response"""
        try:
            # Configure generation parameters
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
            
            # Configure safety settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ]
            
            # Generate content
            return self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
        except Exception as e:
            logger.error(f"Error in content generation: {e}")
            raise
    
    def _is_irrelevant_question(self, text: str) -> bool:
        """
        Check if the response text indicates the question is irrelevant to ministry affairs
        """
        irrelevance_phrases = [
            "unable to answer this question as it is not relevant to the ministry's affairs",
            "not relevant to the ministry's functions",
            "does not fall under the purview of this ministry",
            "outside the scope of this ministry",
            "not within the jurisdiction of this ministry"
        ]
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check if any irrelevance phrase is in the text
        for phrase in irrelevance_phrases:
            if phrase in text_lower:
                return True
        
        return False
    
    def _construct_prompt(
        self, 
        question: str, 
        context: List[Dict[str, Any]], 
        ministry: str
    ) -> str:
        """Construct a prompt for the LLM with question and context"""
        try:
            # Format context into structured sections
            context_parts = []
            
            for i, doc in enumerate(context, 1):
                text = doc.get("text", "").strip()
                metadata = doc.get("metadata", {})
                
                # Extract metadata
                date = metadata.get("date", "Unknown date")
                session = metadata.get("session", "Unknown session")
                source = metadata.get("filename", "Unknown source")
                
                # Format context entry
                context_entry = f"""SOURCE {i}:
Date: {date}
Session: {session}
Source: {source}
Content: {text}
"""
                context_parts.append(context_entry)
            
            # Join all context entries
            context_text = "\n---\n".join(context_parts)
            
            # Create prompt
            prompt = f"""
You are an official representative of the {ministry} in the Indian Parliament.

USER QUESTION:
{question}

CONTEXT FROM PARLIAMENTARY RECORDS:
{context_text}

INSTRUCTIONS:
1. RELEVANCE CHECK:
   * Answer only if the question relates to {ministry}'s functions, policies, or responsibilities.
   * If the question is off-topic, respond: "I am unable to answer this question as it is not relevant to the ministry's affairs."

2. USING CONTEXT:
   * Base your answer primarily on the parliamentary records provided in the context.
   * If the context contains relevant information, cite it specifically (e.g., "According to the record from [date/session]...").
   * If the context is insufficient but the question is valid, use your knowledge of Indian government policies and programs.
   * If using general knowledge, clearly state: "Based on general information about the ministry's policies..."

3. ANSWER FORMAT:
   * Begin with a formal answer to the question.
   * Include specific facts, figures, and dates from the context when available.
   * Organize information logically with clear sections.
   * End with any relevant initiatives or future plans mentioned in the context.

4. TONE:
   * Formal and professional
   * Factual and precise
   * Solution-oriented

Generate a comprehensive, accurate response based on these instructions.
Do not answer irrelevant questions like whats the climate,etc.
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error constructing prompt: {e}")
            # Fallback to simple prompt
            return f"You are representing {ministry}. Answer this question based on the provided context: {question}"
    
    def _format_response(self, text: str, context: List[Dict[str, Any]]) -> str:
        """Format the response with citations and references"""
        try:
            # Clean up the response
            formatted_text = text.strip()
            
            # Check if the question is irrelevant
            is_irrelevant = self._is_irrelevant_question(formatted_text)
            
            # Only add references and metadata if the question is relevant
            if context and not is_irrelevant:
                # Extract sources for citation only if the question is relevant
                sources = []
                for i, doc in enumerate(context[:3], 1):  # Top 3 sources
                    metadata = doc.get("metadata", {})
                    date = metadata.get("date", "Unknown date")
                    session = metadata.get("session", "Unknown session")
                    
                    sources.append(f"[{i}] Parliamentary record from Session {session}, dated {date}")
                
            return formatted_text
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return text