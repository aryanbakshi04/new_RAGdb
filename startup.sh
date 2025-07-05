#!/bin/bash

# Log startup process
echo "Starting application at $(date)" > /home/LogFiles/app.log 2>&1

# Activate virtual environment
source /antenv/bin/activate

# Install dependencies (log any errors)
echo "Upgrading pip..." >> /home/LogFiles/app.log 2>&1
pip install --upgrade pip >> /home/LogFiles/app.log 2>&1
echo "Installing requirements..." >> /home/LogFiles/app.log 2>&1
pip install -r requirements.txt >> /home/LogFiles/app.log 2>&1

# Create data directories in persistent storage
mkdir -p /home/data/vector_db
mkdir -p /home/data/pdf_cache

# Set environment variables for Streamlit
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export STREAMLIT_SERVER_PORT=8000
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_HEADLESS=true

# Start Streamlit with logging
echo "Starting Streamlit app..." >> /home/LogFiles/app.log 2>&1
streamlit run app.py --server.port 8000 --server.enableCORS false --server.headless true >> /home/LogFiles/app.log 2>&1