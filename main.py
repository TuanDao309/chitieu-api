"""
main.py — FastAPI server cho Chi Tiêu Thông Minh
Endpoints:
  POST /parse        — phân tích text (single hoặc batch)
  POST /voice        — nhận audio bytes → PhoWhisper → parse
  POST /correct      — user correction → retrain
  GET  /categories   — danh sách categories
  GET  /health       — health check
"""

import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from expense_engine import (
    ExpenseEngine,
    split_multi_transaction,
    _clean_raw_transcript,
    parse_voice_scheme,
    ALL_CATEGORIES,
)

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(title="Chi Tiêu Thông Minh API", version="2.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # v0.dev, localhost, etc.
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load engine 1 lần khi server khởi động ────────────────────────
engine = ExpenseEngine()

# ── Load PhoWhisper 1 lần (lazy, chỉ khi cần) ────────────────────
_asr_pipe = None

def get_asr_pipe():
    global _asr_pipe
    if _asr_pipe is None:
        if not HF_AVAILABLE:
            raise HTTPException(503, "transformers chưa cài")
        from transformers import pipeline as hf_pipeline
        _asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-small",
            device=-1,
            model_kwargs={"use_safetensors": False},
        )
    return _asr_pipe

try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════
class ParseRequest(BaseModel):
    text: str
    user_id: Optional[str] = "default"

class CorrectRequest(BaseModel):
    text: str
    category: str
    user_id: Optional[str] = "default"


# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.5"}


@app.get("/categories")
def categories():
    return {"categories": ALL_CATEGORIES}


@app.post("/parse")
def parse(req: ParseRequest):
    """
    Phân tích text → category, amount, date, confidence.
    Tự động detect batch nếu có dấu ; hoặc \\ .

    Request:  { "text": "cafe 50k; grab 30k", "user_id": "u123" }
    Response: { "results": [...], "is_batch": true, "total_amount": 80000 }
    """
    text  = req.text.strip()
    parts = split_multi_transaction(text)

    results = []
    for part in parts:
        r = engine.parse(part)
        results.append({
            "text":       r["original_text"],
            "amount":     r["amount"],
            "category":   r["category"],
            "confidence": round(r["confidence"], 4),
            "method":     r["method"],
            "date":       r["date"],
            "top3":       [(c, round(s, 4)) for c, s in r["top3"]],
        })

    return {
        "results":      results,
        "is_batch":     len(parts) > 1,
        "total_amount": sum(r["amount"] for r in results),
    }


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    """
    Nhận file audio WAV (float32, 16kHz mono) → transcribe → parse.

    FE gửi:  FormData với field "file" là audio blob
    Response: kết quả parse giống /parse
    """
    pipe = get_asr_pipe()

    audio_bytes = await file.read()
    # Float32 raw bytes (từ Web Audio API hoặc recording library)
    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

    if audio_np.size == 0:
        raise HTTPException(400, "File audio rỗng")

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    if rms < 0.002:
        raise HTTPException(400, "Âm thanh quá nhỏ, hãy nói to hơn")

    result   = pipe({"array": audio_np, "sampling_rate": 16_000})
    raw_text = result.get("text", "").strip()

    if not raw_text:
        raise HTTPException(400, "Không nhận dạng được giọng nói")

    clean_text  = _clean_raw_transcript(raw_text)
    parsed_text = parse_voice_scheme(clean_text)["text"]
    r = engine.parse(parsed_text)

    return {
        "raw_transcript":   raw_text,
        "clean_transcript": clean_text,
        "text":       r["original_text"],
        "amount":     r["amount"],
        "category":   r["category"],
        "confidence": round(r["confidence"], 4),
        "method":     r["method"],
        "date":       r["date"],
        "top3":       [(c, round(s, 4)) for c, s in r["top3"]],
    }


@app.post("/correct")
def correct(req: CorrectRequest):
    """
    User sửa category sai → model học lại.

    Request:  { "text": "cafe 50k", "category": "ăn uống", "user_id": "u123" }
    Response: { "ok": true }
    """
    if req.category not in ALL_CATEGORIES:
        raise HTTPException(400, f"Category không hợp lệ: {req.category}")

    engine.retrain_with_correction(req.text, req.category)
    return {"ok": True, "text": req.text, "category": req.category}
