FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir     fastapi==0.115.0     uvicorn==0.30.6     transformers==4.44.2     "torch==2.4.1+cpu"     pandas==2.2.2     pydantic==2.8.2     --extra-index-url https://download.pytorch.org/whl/cpu

RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification;     AutoTokenizer.from_pretrained('prajjwal1/bert-tiny');     AutoModelForSequenceClassification.from_pretrained('prajjwal1/bert-tiny', num_labels=4);     print('bert-tiny cached successfully')"

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
