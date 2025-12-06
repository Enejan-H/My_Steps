#!/bin/bash
# Start FastAPI
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
