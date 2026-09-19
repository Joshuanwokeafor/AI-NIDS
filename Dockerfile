# Full-fidelity deployment: runs the COMPLETE three-model hybrid system
# (SVM + Random Forest + LSTM + fuzzy aggregation), unlike the Vercel
# serverless deployment which cannot host TensorFlow. Use this on any
# container platform (Render, Railway, Fly.io, Google Cloud Run, a VPS)
# when you need live inference, retraining, or a working demo of the
# full artifact.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    OPENBLAS_CORETYPE=Nehalem

WORKDIR /app

# libpcap is needed only if you intend to use PCAP/live capture inside
# the container; gcc covers source builds for any wheel-less dependency.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "psycopg[binary]==3.1.19" gunicorn==22.0.0

COPY . .

EXPOSE 8000

# Single worker with a generous timeout: model artifacts are loaded into
# memory per worker, and inference on the full ensemble is CPU-bound.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", \
     "--timeout", "180", "web.app:app"]
