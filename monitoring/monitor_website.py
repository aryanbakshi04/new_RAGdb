import requests
import hashlib
import os
import json
import logging
import subprocess
from datetime import datetime
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='website_monitor.log'
)

WEBSITE_URL = "https://sansad.in/api_ls/question/getFilteredQuestionsAns"  # Replace with actual Sansad website URL
HASH_FILE_PATH = "website_hash.json"
MAX_RETRIES = 3

def get_web_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(Max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
    
    return none

def gen_webhash(content):
    