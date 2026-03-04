FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]