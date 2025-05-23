# Dockerfile actualizado
FROM python:3.11-slim

WORKDIR /app

COPY Docker/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY artifacts ./artifacts
COPY Models ./Models
COPY Experiments ./Experiments
COPY Data ./Data

CMD ["streamlit", "run", "app/streamlit.py", "--server.port=8501", "--server.enableCORS=false"]