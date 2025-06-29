import os
from azure.storage.blob import BlobServiceClient
import tempfile
import shutil
import logging
import chromadb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("azure_storage")

def get_vector_db_from_azure():
    """
    Download vector database files from Azure Blob Storage to a temporary directory.
    
    Returns:
        str: Path to the temporary directory containing the vector database.
    """
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "vector-database"
    temp_dir = tempfile.mkdtemp()
    
    if not connection_string:
        logger.error("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
    
    try:
        logger.info(f"Connecting to Azure Storage using connection string")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        logger.info(f"Downloading vector database files to {temp_dir}")
        blob_list = container_client.list_blobs()
        
        for blob in blob_list:
            blob_client = container_client.get_blob_client(blob.name)
            local_path = os.path.join(temp_dir, blob.name)
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download blob
            with open(local_path, "wb") as download_file:
                download_data = blob_client.download_blob()
                download_file.write(download_data.readall())
            
            logger.info(f"Downloaded {blob.name} to {local_path}")
        
        logger.info(f"Vector database downloaded successfully to {temp_dir}")
        return temp_dir
        
    except Exception as e:
        logger.error(f"Error downloading vector DB: {e}")
        shutil.rmtree(temp_dir)
        raise

def initialize_chroma_client():
    """
    Initialize ChromaDB client using files downloaded from Azure.
    
    Returns:
        tuple: (chromadb.PersistentClient, str) - The ChromaDB client and the temp directory path
    """
    vector_db_path = get_vector_db_from_azure()
    logger.info(f"Initializing ChromaDB client with path: {vector_db_path}")
    
    try:
        client = chromadb.PersistentClient(path=vector_db_path)
        logger.info("ChromaDB client initialized successfully")
        return client, vector_db_path
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {e}")
        shutil.rmtree(vector_db_path)
        raise