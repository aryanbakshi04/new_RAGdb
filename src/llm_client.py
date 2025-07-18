import logging
import google.generativeai as genai
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)

            self.model = genai.GenerativeModel("gemini-2.0-flash-exp")

            self.executor = ThreadPoolExecutor(max_workers=1)

            logger.info("Successfully initialized LLM client")
        except Exception as e:
            logger.error(f"Error initializing LLM client: {e}")
            raise

    def generate_response_sync(
        self, question: str, context: List[Dict[str, Any]], ministry: str
    ) -> str:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                self.generate_response(question, context, ministry)
            )
            loop.close()
            return response
        except Exception as e:
            logger.error(f"Error in synchronous response generation: {e}")
            return (
                "I apologize, but I encountered an error while generating the response. "
                "Please try again with a simpler question or wait a moment before retrying."
            )

    async def generate_response(
        self, question: str, context: List[Dict[str, Any]], ministry: str
    ) -> str:
        try:
            prompt = self._construct_prompt(question, context, ministry)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor, self._generate_content, prompt
            )

            if not response or not response.text:
                logger.warning("Empty response from LLM")
                return (
                    "I apologize, but I couldn't generate a meaningful response. "
                    "Please try rephrasing your question."
                )

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
        try:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]

            return self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

        except Exception as e:
            logger.error(f"Error in content generation: {e}")
            raise

    def _is_irrelevant_question(self, text: str) -> bool:
        irrelevance_phrases = [
            "unable to answer this question as it is not relevant to the ministry's affairs",
            "not relevant to the ministry's functions",
            "does not fall under the purview of this ministry",
            "outside the scope of this ministry",
            "not within the jurisdiction of this ministry",
        ]

        text_lower = text.lower()

        for phrase in irrelevance_phrases:
            if phrase in text_lower:
                return True

        return False

    def _construct_prompt(
        self, question: str, context: List[Dict[str, Any]], ministry: str
    ) -> str:
        try:
            context_parts = []

            for i, doc in enumerate(context, 1):
                text = doc.get("text", "").strip()
                metadata = doc.get("metadata", {})

                date = metadata.get("date", "Unknown date")
                session = metadata.get("session", "Unknown session")
                source = metadata.get("filename", "Unknown source")

                context_entry = f"""SOURCE {i}:
                                Date: {date}
                                Session: {session}
                                Source: {source}
                                Content: {text}
                                """
                context_parts.append(context_entry)

            context_text = "\n---\n".join(context_parts)

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
            return f"You are representing {ministry}. Answer this question based on the provided context: {question}"

    def _format_response(self, text: str, context: List[Dict[str, Any]]) -> str:
        try:
            formatted_text = text.strip()

            is_irrelevant = self._is_irrelevant_question(formatted_text)

            if context and not is_irrelevant:
                sources = []
                for i, doc in enumerate(context[:3], 1):  # Top 3 sources
                    metadata = doc.get("metadata", {})
                    date = metadata.get("date", "Unknown date")
                    session = metadata.get("session", "Unknown session")

                    sources.append(
                        f"[{i}] Parliamentary record from Session {session}, dated {date}"
                    )

            return formatted_text

        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            return text
