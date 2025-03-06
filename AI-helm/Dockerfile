FROM python:3.9-slim
# Lightweight

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

RUN chmod -R a+rwx /app

EXPOSE 5000

CMD ["python", "test.py"]