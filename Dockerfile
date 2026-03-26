FROM python:3.11-slim

WORKDIR /app

# ── Cài deps hệ thống + Ollama ────────────────────────────────────
RUN apt-get update && \
    apt-get install -y curl zstd && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Cài Python deps ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────
COPY expense_engine.py .
COPY main.py .
COPY start.sh .
RUN chmod +x start.sh
RUN mkdir -p models

# ── Pull Qwen2.5:7b khi build → cache vào image ───────────────────
RUN ollama serve & sleep 8 && ollama pull qwen2.5:7b && pkill ollama || true

CMD ["./start.sh"]
