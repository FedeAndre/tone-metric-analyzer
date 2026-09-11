FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AUDIVERIS_VERSION=5.10.2
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUDIVERIS_CMD=/opt/audiveris/bin/Audiveris

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl python3 python3-pip python3-venv \
    && curl -fL --retry 4 --retry-delay 3 \
       -o /tmp/audiveris.deb \
       "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb" \
    && apt-get install -y --no-install-recommends /tmp/audiveris.deb \
    && rm -f /tmp/audiveris.deb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt
COPY . /app
RUN mkdir -p /app/.tone_metric_cache

EXPOSE 8080
CMD ["sh", "-c", "python3 -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
