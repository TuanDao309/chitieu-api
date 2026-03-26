FROM python:3.11-slim

WORKDIR /app

# ── Cài Ollama ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Cài Python deps ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source ───────────────────────────────────────────────────
COPY expense_engine.py .
COPY main.py .
RUN mkdir -p models

# ── Pull Qwen2.5:7b khi build (cache vào image luôn) ─────────────
# Dùng 7b vì Railway Pro có 24GB RAM
RUN ollama serve & sleep 5 && ollama pull qwen2.5:7b && pkill ollama || true

# ── Startup script: chạy Ollama daemon + FastAPI ─────────────────
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
