FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates coreutils curl python3 python3-pip \
    tesseract-ocr tesseract-ocr-eng fontconfig fonts-dejavu-core fonts-liberation \
    libasound2t64 libgtk-3-0t64 libx11-6 libxext6 libxi6 libxrender1 libxtst6 xdg-utils \
 && (curl -fL --retry 8 --retry-delay 5 --retry-all-errors -o /tmp/audiveris.deb \
      "https://github.com/Audiveris/audiveris/releases/download/5.11.0/Audiveris-5.11.0-ubuntu24.04-x86_64.deb" \
     || curl -fL --retry 8 --retry-delay 5 --retry-all-errors -H 'Accept: application/octet-stream' -o /tmp/audiveris.deb \
      "https://api.github.com/repos/Audiveris/audiveris/releases/assets/473797293") \
 && echo "f20113aaa33b3149ec8d6a09b2a7963360e65fafd92d69389987a85bbc3ec7a3  /tmp/audiveris.deb" | sha256sum -c - \
 && dpkg-deb -x /tmp/audiveris.deb / && test -x /opt/audiveris/bin/Audiveris \
 && rm -f /tmp/audiveris.deb && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt
COPY app.py core.py hit_model.py hit_geometry.py hit_align.py render.py ./
COPY tests ./tests
RUN ! grep -R -n -E 'isotonic|pool.adjacent|PAVA|DISPLAY_MIN_GAP|build_hits_from_canonical|canonical_score|_cluster_chords_by_x|DISPLACED_MIN_DX|x_overlaps|reconciled-hit-engine' /app/app.py /app/core.py /app/hit_model.py /app/hit_geometry.py /app/hit_align.py /app/render.py /app/tests \
 && python3 -m py_compile app.py core.py hit_model.py hit_geometry.py hit_align.py render.py tests/test_hits.py \
 && PYTHONPATH=/app python3 -m unittest discover -s /app/tests -v
EXPOSE 8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
