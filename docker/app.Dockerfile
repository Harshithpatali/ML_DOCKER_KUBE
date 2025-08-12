FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts
COPY models/ ./models

WORKDIR /app/scripts

EXPOSE 5000

CMD ["python", "app.py"]
