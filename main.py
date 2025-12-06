import subprocess

# Start FastAPI in background
fastapi_proc = subprocess.Popen(
    ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
)

# Start Streamlit in foreground
subprocess.run(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
)

# When Streamlit exits, stop FastAPI
fastapi_proc.terminate()