FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AUDIVERIS_VERSION=5.10.2
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUDIVERIS_CMD=/opt/audiveris/bin/Audiveris \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates coreutils curl unzip python3 python3-pip python3-venv \
       fontconfig fonts-dejavu-core fonts-liberation \
       tesseract-ocr tesseract-ocr-eng \
       libasound2t64 libx11-6 libxext6 libxi6 libxrender1 libxtst6 xdg-utils \
    && curl -fL --retry 4 --retry-delay 3 \
       -o /tmp/audiveris.deb \
       "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb" \
    && dpkg-deb -x /tmp/audiveris.deb / \
    && test -x /opt/audiveris/bin/Audiveris \
    && mkdir -p /root/.config/AudiverisLtd/audiveris/tessdata \
    && ENG="$(dpkg -L tesseract-ocr-eng | grep '/eng.traineddata$' | head -n1)" \
    && test -n "$ENG" \
    && cp "$ENG" /root/.config/AudiverisLtd/audiveris/tessdata/eng.traineddata \
    && fc-cache -f \
    && rm -f /tmp/audiveris.deb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /transport
COPY runtime.part*.b64 /transport/
RUN test "$(find /transport -maxdepth 1 -type f -name 'runtime.part*.b64' | wc -l)" -eq 28 \
    && cat /transport/runtime.part*.b64 | base64 -d > /tmp/tone_metric_runtime_source.zip \
    && echo "88fc9e687de8a2d576021d3202224a2166975538870cf0ce0be861a358e360a2  /tmp/tone_metric_runtime_source.zip" | sha256sum -c -

WORKDIR /app
RUN unzip -q /tmp/tone_metric_runtime_source.zip -d /app \
    && rm -f /tmp/tone_metric_runtime_source.zip \
    && rm -rf /transport \
    && python3 -m pip install --no-cache-dir --break-system-packages -r /app/requirements.txt \
    && PYTHONDONTWRITEBYTECODE=1 python3 -B /app/validate_release.py \
    && test "$(python3 -B -c 'import app; print(app.app.version)')" = "0.16.4" \
    && test -x /opt/audiveris/bin/Audiveris

EXPOSE 8080
CMD ["sh", "-c", "python3 -B -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
