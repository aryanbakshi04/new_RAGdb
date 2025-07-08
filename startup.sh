#!/bin/bash
cd /home/site/wwwroot

# Create logs directory if it doesn't exist
mkdir -p logs

# Test Python version
python3 --version > logs/python_version.txt 2>&1
which python3 >> logs/python_version.txt 2>&1

# Install requirements and start app
python3 -m pip install -r requirements.txt > logs/startup.log 2>&1
streamlit run app.py --server.port 8000 --server.address 0.0.0.0 >> logs/startup.log 2>&1