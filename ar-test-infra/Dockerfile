# Built to run somewhere with real network access — unlike the sandbox this
# project was developed in, this environment CAN reach PyPI, so faiss-cpu
# installs cleanly here instead of failing like it did in dev.
FROM python:3.12-slim

WORKDIR /app

# System libs OpenCV needs at runtime (only relevant if you also containerize
# the embedding pipeline — the backend itself doesn't touch OpenCV, but this
# keeps the image ready if you fold glasses_client's server-side pieces in later).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir faiss-cpu

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
