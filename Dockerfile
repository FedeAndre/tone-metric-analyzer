FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY app.py verify_fixture.py ./
COPY fixtures ./fixtures

# Download HOMR's own ONNX models into the image, then run a real OMR inference.
RUN homr --init
RUN homr --no-title /app/fixtures/tabi.jpg && python /app/verify_fixture.py

EXPOSE 8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
