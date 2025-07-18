
import streamlit as st
import os
import logging
from pathlib import Path
import sys
import asyncio
import base64

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import Config
from src.vector_store import VectorStore
from src.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("streamlit_app")

def run_async(coro):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coro)
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in async execution: {e}")
        return f"Error: {str(e)}"

def is_irrelevant_question(response):
    irrelevance_phrases = [
        "unable to answer this question as it is not relevant to the ministry's affairs",
        "not relevant to the ministry's functions",
        "does not fall under the purview of this ministry",
        "outside the scope of this ministry",
        "not within the jurisdiction of this ministry",
    ]
    response_lower = response.lower()
    for phrase in irrelevance_phrases:
        if phrase in response_lower:
            return True
    return False

@st.cache_resource
def initialize_components():
    try:
        if not Config.validate_environment():
            st.error("Environment validation failed. Check your API keys.")
            return None, None
        vector_store = VectorStore()
        llm_client = LLMClient()
        return vector_store, llm_client
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        logger.error(f"Error initializing components: {e}")
        return None, None

def main():
    st.set_page_config(
        page_title="Parliamentary Q&A Assistant",
        page_icon="🏛️",
        layout="wide",
    )
    st.title("Parliamentary Ministry Q&A Assistant")
    st.subheader("Ask questions about parliamentary affairs by ministry")
    vector_store, llm_client = initialize_components()
    if not vector_store or not llm_client:
        st.warning("System initialization failed. Please check logs.")
        return
    st.sidebar.title("Select a Ministry")
    indexed_ministries = list(vector_store.indexed_ministries)
    if not indexed_ministries:
        st.warning("No ministries have been indexed. Please run the PDF fetcher first.")
        return
    indexed_ministries.sort()
    selected_ministry = st.sidebar.selectbox("", options=indexed_ministries, index=0)
    query = st.text_input("Enter your question for the selected ministry:", key="query",label_visibility="visible")
    if query and st.button("Submit Question"):
        with st.spinner("Loading"):
            try:
                documents = vector_store.search_by_text(
                    query=query,
                    ministry=selected_ministry,
                    n_results=Config.MAX_DOCS_PER_QUERY,
                )
                if not documents:
                    st.warning(f"No relevant documents found for {selected_ministry}.")
                    return
                with st.spinner("Please wait..."):
                    response = run_async(
                        llm_client.generate_response(
                            question=query,
                            context=documents,
                            ministry=selected_ministry,
                        )
                    )
                    st.markdown("### Response:")
                    st.markdown(response)
                if not is_irrelevant_question(response):
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(documents, 1):
                            st.markdown(f"**Source {i}**")
                            st.markdown(doc["text"])
                            metadata = doc.get("metadata", {})
                            filename = metadata.get("filename", "Unknown")
                            pdf_path = (
                                os.path.join(Config.PDF_CACHE_DIR, filename)
                                if filename != "Unknown"
                                else None
                            )
                            if pdf_path and os.path.exists(pdf_path):
                                try:
                                    with open(pdf_path, "rb") as pdf_file:
                                        pdf_bytes = pdf_file.read()
                                    st.download_button(
                                        label="Download PDF",
                                        data=pdf_bytes,
                                        file_name=filename,
                                        mime="application/pdf",
                                        key=f"download_{i}",
                                    )
                                    pdf_b64 = base64.b64encode(pdf_bytes).decode()
                                    href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="{filename}" target="_blank">View PDF</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                except Exception as e:
                                    st.warning(f"Could not load PDF: {str(e)}")
                            else:
                                st.warning("PDF file not found in cache")
                            st.markdown("---")
                else:
                    st.info(
                        "Alert: This question is not relevant to the ministry affairs."
                    )
            except Exception as e:
                st.error(f"Error processing query: {e}")
                logger.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()