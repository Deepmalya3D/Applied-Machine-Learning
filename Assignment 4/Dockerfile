FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY score.py .
COPY best_model.joblib .

EXPOSE 8000

CMD ["python", "app.py"]