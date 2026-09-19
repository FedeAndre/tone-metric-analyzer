FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps \
       'git+https://github.com/BreezeWhite/oemer@dbe2a933d630d0f74805d717960eb259473f5978'

COPY optical_reader.py rhythm_reconstructor.py tma_analysis.py analyzer.py app.py ./
COPY tone_metric ./tone_metric

# Bake the low-level optical checkpoints into the image so requests never
# download model weights into a live Railway container.
RUN python -c "from optical_reader import ensure_checkpoints; ensure_checkpoints()"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
