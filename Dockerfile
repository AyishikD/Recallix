# Recallix: vector engine (C++) + memory API (Python) in one image.
#   docker compose up -d          (see docker-compose.yml)
#   docker compose logs recallix  (the admin API key is printed once, on first start)

# ---- build the vector engine -------------------------------------------------
FROM debian:bookworm-slim AS engine
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ make && rm -rf /var/lib/apt/lists/*
COPY backend/infra_cpp /src
# portable build: no -march tuning, so the binary runs on any x86-64 / arm64 host
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DRECALLIX_DISABLE_AVX2=ON \
 && cmake --build /build --target memory_service -j"$(nproc)"

# ---- API ---------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 \
    RECALLIX_STORAGE_ROOT=/data RECALLIX_EMBEDDING_PROVIDER=local \
    RECALLIX_ENGINE_URL=http://127.0.0.1:8080 RECALLIX_ENGINE_HOST=127.0.0.1 RECALLIX_ENGINE_PORT=8080 \
    RECALLIX_API_HOST=0.0.0.0 RECALLIX_API_PORT=8000 HF_HOME=/data/models
WORKDIR /app
# CPU-only torch keeps the image ~1.5 GB instead of ~5 GB
COPY requirements.txt requirements-local.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-local.txt
COPY --from=engine /build/memory_service /usr/local/bin/memory_service
COPY backend ./backend
COPY server.py config.yaml ./
COPY deploy/entrypoint.sh /usr/local/bin/recallix-entrypoint
RUN rm -rf backend/infra_cpp/build backend/storage && chmod +x /usr/local/bin/recallix-entrypoint \
 && useradd --create-home --uid 10001 recallix && mkdir -p /data && chown recallix /data
USER recallix
VOLUME /data
EXPOSE 8000
# /health is 200 only when the API and the vector engine both answer (503 otherwise, which urlopen raises on)
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
ENTRYPOINT ["recallix-entrypoint"]
