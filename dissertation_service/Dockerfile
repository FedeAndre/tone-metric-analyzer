FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AUDIVERIS_VERSION=5.10.2
ARG RAILWAY_GIT_COMMIT_SHA
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUDIVERIS_CMD=/opt/audiveris/bin/Audiveris

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl git python3 python3-pip python3-venv \
       libasound2t64 libx11-6 libxext6 libxi6 libxrender1 libxtst6 xdg-utils \
       fontconfig fonts-dejavu-core fonts-liberation \
       tesseract-ocr tesseract-ocr-eng \
    && fc-cache -f \
    && fc-match "DejaVu Sans" | grep -q "DejaVuSans" \
    && mkdir -p /root/.config/AudiverisLtd/audiveris/tessdata \
    && cp -f /usr/share/tesseract-ocr/5/tessdata/*.traineddata /root/.config/AudiverisLtd/audiveris/tessdata/ \
    && test -s /root/.config/AudiverisLtd/audiveris/tessdata/eng.traineddata \
    && curl -fL --retry 4 --retry-delay 3 \
       -o /tmp/audiveris.deb \
       "https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb" \
    && dpkg-deb -x /tmp/audiveris.deb / \
    && test -x /opt/audiveris/bin/Audiveris \
    && rm -f /tmp/audiveris.deb \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/FedeAndre/tone-metric-analyzer.git /app \
    && if [ -n "${RAILWAY_GIT_COMMIT_SHA}" ]; then \
         git -C /app checkout --detach "${RAILWAY_GIT_COMMIT_SHA}"; \
       else \
         git -C /app checkout main; \
       fi \
    && echo "Building repository revision $(git -C /app rev-parse HEAD)"

WORKDIR /app
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Preserve and re-run every validated Level-1 regression before Level 2.
# v0.6 imports the frozen v0.4 Level-1 analysis and v0.5 exact OMR registration,
# then adds only the recursive dissertation Level-2 layer.
RUN python3 validate_level1_clean_v4.py
RUN python3 validate_level1_clean_v5.py
RUN python3 validate_level1_level2_clean_v6.py
RUN python3 -c "import level1_level2_clean_v6; print(level1_level2_clean_v6.APP_VERSION)"

EXPOSE 8080
CMD ["python3", "-m", "uvicorn", "--app-dir", "/app", "level1_level2_clean_v6:app", "--host", "0.0.0.0", "--port", "8080"]
