FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AUDIVERIS_VERSION=5.10.2
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUDIVERIS_CMD=/opt/audiveris/bin/Audiveris

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl unzip python3 python3-pip python3-venv \
       libasound2t64 libx11-6 libxext6 libxi6 libxrender1 libxtst6 xdg-utils \
    && curl -fL --retry 4 --retry-delay 3 \
       -o /tmp/audiveris.deb \
       "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb" \
    && dpkg-deb -x /tmp/audiveris.deb / \
    && test -x /opt/audiveris/bin/Audiveris \
    && rm -f /tmp/audiveris.deb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY tone_metric_runtime_source.zip /tmp/tone_metric_runtime_source.zip
RUN unzip -q /tmp/tone_metric_runtime_source.zip -d /app \
    && rm -f /tmp/tone_metric_runtime_source.zip \
    && mkdir -p /app/.tone_metric_cache
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

EXPOSE 8080
CMD ["sh", "-c", "python3 -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
