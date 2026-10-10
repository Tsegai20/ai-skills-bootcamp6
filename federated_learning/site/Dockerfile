FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model so container starts fast
RUN python -c "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification; \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased'); \
    DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)"

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
