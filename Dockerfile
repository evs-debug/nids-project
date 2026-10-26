FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN sed -i 's/^xgboost$/xgboost-cpu/' requirements.txt && pip install -r requirements.txt
COPY src/ src/
COPY server/ server/
RUN python src/get_data.py
# train once at build time so a restart takes seconds, not a minute
RUN python -c "import sys; sys.path.insert(0, 'src'); from server.engine import Engine; Engine()"
COPY --from=web /web/dist web/dist
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
