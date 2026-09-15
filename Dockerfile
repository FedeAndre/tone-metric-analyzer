FROM eclipse-temurin:25-jdk AS audiveris-builder
ARG AUDIVERIS_VERSION=5.11.0
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${AUDIVERIS_VERSION}" https://github.com/Audiveris/audiveris.git /src/audiveris \
    && sed -i 's/new MorphoProcessor(se)\.close(buffer);/new MorphoProcessor(se).fclose(buffer);/' \
       /src/audiveris/app/src/main/java/org/audiveris/omr/sheet/beam/SpotsBuilder.java \
    && grep -F 'new MorphoProcessor(se).fclose(buffer);' \
       /src/audiveris/app/src/main/java/org/audiveris/omr/sheet/beam/SpotsBuilder.java
WORKDIR /src/audiveris
RUN ./gradlew --no-daemon -PisFlatpak=true :app:installDist -x test \
    && test -x /src/audiveris/app/build/install/Audiveris/bin/Audiveris

FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:${PATH} \
    AUDIVERIS_CMD=/opt/audiveris/bin/Audiveris \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates coreutils python3 python3-pip python3-venv \
       fontconfig fonts-dejavu-core fonts-liberation \
       tesseract-ocr tesseract-ocr-eng \
       libasound2t64 libgtk-3-0t64 libx11-6 libxext6 libxi6 libxrender1 libxtst6 xdg-utils \
    && mkdir -p /root/.config/AudiverisLtd/audiveris/tessdata \
    && ENG="$(dpkg -L tesseract-ocr-eng | grep '/eng.traineddata$' | head -n1)" \
    && test -n "$ENG" \
    && cp "$ENG" /root/.config/AudiverisLtd/audiveris/tessdata/eng.traineddata \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*
COPY --from=audiveris-builder /opt/java/openjdk /opt/java/openjdk
COPY --from=audiveris-builder /src/audiveris/app/build/install/Audiveris /opt/audiveris
RUN test -x /opt/audiveris/bin/Audiveris \
    && java -version
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
