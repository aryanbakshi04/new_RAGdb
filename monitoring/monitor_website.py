import requests
import hashlib
import os
import json
import logging
import subprocess
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='website_monitor.log'
)

WEBSITE_URL = "https://sansad.in/api_ls/question/qetFilteredQuestionsAns"  # Use the correct endpoint
HASH_FILE_PATH = "website_hash.json"
MAX_RETRIES = 3

def get_web_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
    return None

def gen_webhash(content):
    """Generate a SHA256 hash from the website content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def load_previous_hash():
    if os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, 'r') as f:
            data = json.load(f)
            return data.get('hash')
    return None

def save_current_hash(hash_value):
    with open(HASH_FILE_PATH, 'w') as f:
        json.dump({'hash': hash_value, 'timestamp': datetime.now().isoformat()}, f)

def main():
    try:
        content = get_web_content(WEBSITE_URL)
    except Exception as e:
        logging.error(f"Failed to fetch website content: {e}")
        return

    current_hash = gen_webhash(content)
    previous_hash = load_previous_hash()

    if previous_hash != current_hash:
        logging.info("Website content has changed. Fetching new PDFs and updating database incrementally...")
        try:
            # Fetch new PDFs (from new_pdf_urls.json or similar)
            subprocess.run(["python", "src/fetch_ministry_pdfs.py", "--url-file", "new_pdf_urls.json"], check=True)
            # Only update database for new PDFs
            subprocess.run(["python", "src/update_ministry_database.py", "--url-file", "new_pdf_urls.json"], check=True)
            logging.info("Successfully fetched new PDFs and incrementally updated the database.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running update scripts: {e}")
        save_current_hash(current_hash)
    else:
        logging.info("No change detected on website.")

if __name__ == "__main__":
    main()