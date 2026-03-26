#!/bin/bash
# Chạy Ollama daemon nền trước
ollama serve &
OLLAMA_PID=$!

# Đợi Ollama sẵn sàng
echo "⏳ Chờ Ollama khởi động..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama ready!"
        break
    fi
    sleep 1
done

# Chạy FastAPI
echo "🚀 Khởi động FastAPI..."
python -c "
import os, uvicorn
uvicorn.run('main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), workers=1)
"
