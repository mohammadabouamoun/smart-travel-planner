# ========== BUILDER STAGE ==========
FROM python:3.12 AS builder

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy the locked requirements and install
COPY requirements.lock .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache --prefix=/install -r requirements.lock

# ========== FINAL STAGE ==========
FROM python:3.12-slim

WORKDIR /app

# Runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev libopenblas-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ ./backend/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]