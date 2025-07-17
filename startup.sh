cd /home/site/wwwroot

mkdir -p logs

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade pysqlite3-binary

export PYTHONPATH=/home/site/wwwroot/antenv/lib/python3.11/site-packages/pysqlite3
export LD_PRELOAD=/home/site/wwwroot/antenv/lib/python3.11/site-packages/pysqlite3/libsqlite3.so

streamlit run app.py --server.port 8000 --server.address 0.0.0.0