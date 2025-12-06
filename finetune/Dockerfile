# Base image
FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Önce requirements kopyala ve yükle
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala
COPY app.py api.py train.py inference.py main.py ./

# Modeli container içinde indirmek daha güvenli
# Eğer önceden kaydedilmiş model varsa:
COPY finetuned_model_final ./finetuned_model_final


# Copy start script
COPY start.sh .
RUN chmod +x start.sh

# Expose ports
EXPOSE 8000 8501

# Start both FastAPI and Streamlit
CMD ["./start.sh"]