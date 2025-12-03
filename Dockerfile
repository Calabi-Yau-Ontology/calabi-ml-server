FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (필요하면 추가)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
  && apt-get update \
  && apt-get -y install libgl1-mesa-glx \
  && apt-get -y install libglib2.0-0 \
  && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .env ./.env

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]