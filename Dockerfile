FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

# RapidOCR currently pulls the GUI OpenCV wheel even though HOMR itself uses
# opencv-python-headless. Supply only the shared libraries needed to import it.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libxcb1 libx11-6 libxext6 libxrender1 libsm6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY app.py verify_fixture.py ./
COPY fixtures ./fixtures

# Download HOMR's own ONNX models into the image, then run a real OMR inference.
RUN homr --init
RUN homr /app/fixtures/tabi.jpg && python /app/verify_fixture.py

EXPOSE 8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
