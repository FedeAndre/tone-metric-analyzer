FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG AUDIVERIS_VERSION=5.11.0
ARG AUDIVERIS_ASSET_ID=473797293
ARG AUDIVERIS_SHA256=f20113aaa33b3149ec8d6a09b2a7963360e65fafd92d69389987a85bbc3ec7a3
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUDIVERIS_CMD=/opt/audiveris/bin/Audiveris \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates coreutils curl unzip python3 python3-pip python3-venv \
       fontconfig fonts-dejavu-core fonts-liberation \
       tesseract-ocr tesseract-ocr-eng \
       libasound2t64 libgtk-3-0t64 libx11-6 libxext6 libxi6 libxrender1 libxtst6 xdg-utils \
    && (curl -fL --retry 8 --retry-delay 5 --retry-all-errors \
          -o /tmp/audiveris.deb \
          "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb" \
        || curl -fL --retry 8 --retry-delay 5 --retry-all-errors \
          -H 'Accept: application/octet-stream' \
          -o /tmp/audiveris.deb \
          "https://api.github.com/repos/Audiveris/audiveris/releases/assets/${AUDIVERIS_ASSET_ID}") \
    && echo "${AUDIVERIS_SHA256}  /tmp/audiveris.deb" | sha256sum -c - \
    && dpkg-deb -x /tmp/audiveris.deb / \
    && test -x /opt/audiveris/bin/Audiveris \
    && mkdir -p /root/.config/AudiverisLtd/audiveris/tessdata \
    && ENG="$(dpkg -L tesseract-ocr-eng | grep '/eng.traineddata$' | head -n1)" \
    && test -n "$ENG" \
    && cp "$ENG" /root/.config/AudiverisLtd/audiveris/tessdata/eng.traineddata \
    && fc-cache -f \
    && rm -f /tmp/audiveris.deb \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt
COPY app.py /app/app.py
COPY tone_metric /app/tone_metric
COPY static /app/static
COPY v0152_sha256.txt /app/v0152_sha256.txt
RUN mkdir -p /app/.tone_metric_cache \
    && cd /app \
    && sha256sum -c v0152_sha256.txt \
    && test ! -e /app/validate_release.py \
    && test ! -e /app/tone_metric/omr_exact.py \
    && test ! -e /app/tone_metric/omr_visual_rhythm.py \
    && test ! -e /app/tone_metric/layer_registration.py \
    && test ! -e /app/tone_metric/render.py \
    && python3 -B -c "import tone_metric; assert tone_metric.__version__ == '0.15.2'" \
    && python3 -m py_compile /app/app.py /app/tone_metric/*.py
EXPOSE 8080
CMD ["sh", "-c", "python3 -B -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
