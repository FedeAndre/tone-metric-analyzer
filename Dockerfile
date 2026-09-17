FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG AUDIVERIS_VERSION=5.11.0
ARG AUDIVERIS_ASSET_ID=473797293
ARG AUDIVERIS_SHA256=f20113aaa33b3149ec8d6a09b2a7963360e65fafd92d69389987a85bbc3ec7a3
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 AUDIVERIS_CMD=/opt/audiveris/bin/Audiveris JAVA_TOOL_OPTIONS=-Djava.awt.headless=true
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates coreutils curl python3 python3-pip \
    tesseract-ocr tesseract-ocr-eng fontconfig fonts-dejavu-core fonts-liberation \
    libasound2t64 libgtk-3-0t64 libx11-6 libxext6 libxi6 libxrender1 libxtst6 xdg-utils \
 && (curl -fL --retry 8 --retry-delay 5 --retry-all-errors -o /tmp/audiveris.deb \
      "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb" \
     || curl -fL --retry 8 --retry-delay 5 --retry-all-errors -H 'Accept: application/octet-stream' \
      -o /tmp/audiveris.deb "https://api.github.com/repos/Audiveris/audiveris/releases/assets/${AUDIVERIS_ASSET_ID}") \
 && echo "${AUDIVERIS_SHA256}  /tmp/audiveris.deb" | sha256sum -c - \
 && dpkg-deb -x /tmp/audiveris.deb / \
 && test -x /opt/audiveris/bin/Audiveris \
 && rm -f /tmp/audiveris.deb && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt
COPY core.py render.py app.py ./
COPY tests ./tests
RUN python3 -m py_compile app.py core.py render.py tests/test_hits.py \
 && PYTHONPATH=/app python3 -m unittest discover -s /app/tests -v
EXPOSE 8080
CMD ["sh","-c","python3 -B -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
