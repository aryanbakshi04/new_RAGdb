#!/bin/bash
cd /home/site/wwwroot

# Create logs directory if it doesn't exist
mkdir -p logs

# Install requirements
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade pysqlite3-binary

# Set environment variables for pysqlite3
export PYTHONPATH=/home/site/wwwroot/antenv/lib/python3.11/site-packages/pysqlite3
export LD_PRELOAD=/home/site/wwwroot/antenv/lib/python3.11/site-packages/pysqlite3/libsqlite3.so

# Start the app
streamlit run app.py --server.port 8000 --server.address 0.0.0.0
