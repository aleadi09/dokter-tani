"""Dokter Tani — halaman uji (Gradio) + API untuk bot Telegram.

Jalan di laptop:      python app/app.py            → buka http://localhost:7860
Jalan di HF Spaces:   file ini menjadi app.py di root Space (lihat scripts/build_space.py)

Endpoint API (untuk n8n / bot):
  GET  /api/health
  POST /api/diagnose      multipart: image=<file>, session_id=<id chat>, [lat, lon]
                          (foto bukan daun → label "bukan_daun", gradcam_jpg_base64 = null)
  POST /api/diagnose_b64  JSON: {"session_id", "image_base64", "lat"?, "lon"?}
  POST /api/ask           JSON: {"session_id", "question"}
  POST /api/reset         JSON: {"session_id"}
Bila environment variable API_TOKEN diisi, setiap panggilan API wajib membawa header  X-API-Key: <API_TOKEN>.

Environment variable lain (Secrets di Hugging Face Space):
  GROQ_API_KEY / GEMINI_API_KEY   kunci LLM (tanpa ini agent memakai mode tanpa LLM)
  LLM_PROVIDER (groq|gemini), LLM_MODEL   opsional
  MODEL_REPO   repo model di Hugging Face Hub (mis. "nama/dokter-tani-cabai"); kosong = pakai folder model lokal
  TELEGRAM_BOT_URL   link bot untuk ditampilkan di halaman
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent           # Space: app.py di root · repo: app/app.py
sys.path.insert(0, str(ROOT))


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_env()

import gradio as gr                                                # noqa: E402
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile   # noqa: E402
from PIL import Image                                              # noqa: E402

from src.agent.agent import DokterTaniAgent                        # noqa: E402
from src.model.leaf_gate import LeafGate                            # noqa: E402
from src.model.predictor import DokterTaniPredictor, display_name  # noqa: E402
from src.rag.knowledge import KnowledgeBase                        # noqa: E402


# ------------------------------------------------------------------ memuat model & knowledge base
def load_predictor() -> DokterTaniPredictor:
    repo = os.environ.get("MODEL_REPO", "").strip()
    if repo:
        return DokterTaniPredictor.from_hub(repo, token=os.environ.get("HF_TOKEN") or None)
    for cand in (ROOT / "model", ROOT / "export" / "hf_model", ROOT / "checkpoints" / "dokter_tani_final.pt"):
        if cand.exists():
            return DokterTaniPredictor(cand)
    raise FileNotFoundError("Model tidak ditemukan: isi MODEL_REPO atau sediakan folder model/.")


print("Memuat model foto...", flush=True)
PREDICTOR = load_predictor()
print("Memuat penjaga gerbang (CLIP: foto daun atau bukan)...", flush=True)
GATE = LeafGate() if os.environ.get("LEAF_GATE", "1") != "0" else None
if GATE and not GATE.enabled:
    print(f"  Gerbang daun nonaktif: {GATE.error}", flush=True)
print("Memuat knowledge base...", flush=True)
KB = KnowledgeBase.load(ROOT / "rag_index")
AGENT = DokterTaniAgent(KB, provider=os.environ.get("LLM_PROVIDER", "groq"),
                        model=os.environ.get("LLM_MODEL") or None, retrieval_mode="hybrid")
LOCK = threading.Lock()          # satu agent dipakai bergantian; state percakapan disimpan per sesi
GATE_ON = bool(GATE and GATE.enabled)
print(f"Siap: {PREDICTOR.arch}, {len(KB.chunks)} chunk, LLM {'aktif: ' + AGENT.model if AGENT.llm_enabled else 'tidak aktif'}, "
      f"gerbang daun {'aktif (batas ' + str(GATE.threshold) + ')' if GATE_ON else 'nonaktif'}", flush=True)

NOT_LEAF_TEXT = ("Maaf, foto ini sepertinya bukan foto daun cabai, jadi belum bisa saya periksa. 🌿\n\n"
                 "Silakan kirim foto daun cabai yang bergejala:\n"
                 "• satu daun, dekat, memenuhi sebagian besar foto\n"
                 "• cahaya terang dan tidak buram\n"
                 "• latar polos bila memungkinkan, misalnya kertas putih")


def check_leaf(img: Image.Image) -> dict:
    return GATE.check(img) if GATE_ON else {"is_leaf": True, "leaf_score": None, "top": None, "gate": "nonaktif"}


def _snapshot() -> dict:
    return {"diag": AGENT.diag, "messages": list(AGENT.messages)}


def _restore(state: dict | None):
    AGENT.reset()
    if state:
        AGENT.diag, AGENT.messages = state.get("diag"), list(state.get("messages") or AGENT.messages)


def run_diagnose(img: Image.Image, state: dict | None = None, location=None) -> tuple[dict, Image.Image, dict, dict]:
    diag = PREDICTOR.predict(img)
    overlay, _ = PREDICTOR.gradcam(img)
    with LOCK:
        AGENT.reset()
        res = AGENT.diagnose(diag, location=location)
        new_state = _snapshot()
    return diag, overlay, res, new_state


def run_ask(question: str, state: dict | None) -> tuple[dict, dict]:
    with LOCK:
        _restore(state)
        res = AGENT.ask(question)
        new_state = _snapshot()
    return res, new_state


# ------------------------------------------------------------------ API untuk bot
api = FastAPI(title="Dokter Tani API")
SESSIONS: dict[str, dict] = {}
SESSION_TTL = 6 * 3600


def _auth(x_api_key: str | None):
    token = os.environ.get("API_TOKEN", "").strip()
    if token and x_api_key != token:
        raise HTTPException(status_code=401, detail="API key salah atau tidak ada (header X-API-Key).")


def _gc_sessions():
    now = time.time()
    for k in [k for k, v in SESSIONS.items() if now - v["t"] > SESSION_TTL]:
        SESSIONS.pop(k, None)


def _to_pil(img) -> Image.Image:
    """Grad-CAM bisa berupa PIL Image atau array numpy (uint8 0–255 / float 0–1)."""
    if isinstance(img, Image.Image):
        return img.copy()
    import numpy as np
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)                                   # CHW → HWC
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def _jpeg_b64(img, max_side: int = 768) -> str:
    img = _to_pil(img)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _diagnose_response(session_id: str, img: Image.Image, location=None) -> dict:
    _gc_sessions()
    gate = check_leaf(img)
    if not gate["is_leaf"]:          # bukan daun: CNN tidak dipakai, riwayat sesi tidak diubah
        print(f"  ditolak gerbang: skor daun {gate['leaf_score']} ({gate['top']})", flush=True)
        return {"text": NOT_LEAF_TEXT, "label": "bukan_daun", "nama": "Bukan foto daun", "prob": 0.0, "top": [],
                "mode": "gerbang daun", "seconds": 0.0, "gradcam_jpg_base64": None,
                "leaf_score": gate["leaf_score"]}
    diag, overlay, res, state = run_diagnose(img, SESSIONS.get(session_id, {}).get("state"), location)
    SESSIONS[session_id] = {"state": state, "t": time.time()}
    return {"text": res["text"], "label": diag["label"], "nama": diag["nama"], "prob": round(diag["prob"], 4),
            "top": [{"nama": t["nama"], "prob": round(t["prob"], 4)} for t in diag["top"]],
            "mode": res["mode"], "seconds": res["seconds"], "gradcam_jpg_base64": _jpeg_b64(overlay),
            "leaf_score": gate["leaf_score"]}


@api.get("/api/health")
def health():
    return {"status": "ok", "model": PREDICTOR.arch, "leaf_gate": GATE.threshold if GATE_ON else None, "classes": PREDICTOR.classes, "chunks": len(KB.chunks),
            "llm": AGENT.model if AGENT.llm_enabled else None, "sessions": len(SESSIONS)}


@api.post("/api/diagnose")
async def diagnose_upload(image: UploadFile = File(...), session_id: str = Form(...), lat: float | None = Form(None),
                          lon: float | None = Form(None), x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")
    loc = (lat, lon) if lat is not None and lon is not None else None
    return _diagnose_response(session_id, img, loc)


@api.post("/api/diagnose_b64")
def diagnose_b64(payload: dict = Body(...), x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    try:
        img = Image.open(io.BytesIO(base64.b64decode(payload["image_base64"].split(",")[-1])))
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 bukan gambar yang valid.")
    lat, lon = payload.get("lat"), payload.get("lon")
    return _diagnose_response(str(payload.get("session_id", "default")), img,
                              (lat, lon) if lat is not None and lon is not None else None)


@api.post("/api/ask")
def ask(payload: dict = Body(...), x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    sid, question = str(payload.get("session_id", "default")), str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="question kosong.")
    _gc_sessions()
    res, state = run_ask(question, SESSIONS.get(sid, {}).get("state"))
    SESSIONS[sid] = {"state": state, "t": time.time()}
    return {"text": res["text"], "mode": res["mode"], "seconds": res["seconds"],
            "has_diagnosis": bool(state.get("diag"))}


@api.post("/api/reset")
def reset(payload: dict = Body(...), x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    SESSIONS.pop(str(payload.get("session_id", "default")), None)
    return {"status": "ok"}


# ------------------------------------------------------------------ halaman uji (Gradio)
def _md(text: str) -> str:
    """Pertahankan baris baru (poin •, langkah) saat ditampilkan sebagai Markdown."""
    return text.replace("\n", "  \n")


def ui_analyze(img, state):
    if img is None:
        return None, None, "Silakan upload atau ambil foto daun cabai dulu.", [], state
    gate = check_leaf(img)
    if not gate["is_leaf"]:
        return ({"Bukan foto daun": 1.0}, None,
                _md(NOT_LEAF_TEXT) + f"\n\n---\n*gerbang daun: skor {gate['leaf_score']}*", [], state)
    diag, overlay, res, state = run_diagnose(img, state)
    probs = {t["nama"]: float(t["prob"]) for t in diag["top"]}
    footer = f"\n\n---\n*mode: {res['mode']} · {res['seconds']} detik*"
    return probs, overlay, _md(res["text"]) + footer, [], state


def ui_chat(message, history, state):
    message = (message or "").strip()
    if not message:
        return history, "", state
    res, state = run_ask(message, state)
    history = (history or []) + [{"role": "user", "content": message}, {"role": "assistant", "content": _md(res["text"])}]
    return history, "", state


def build_ui() -> gr.Blocks:
    bot_url = os.environ.get("TELEGRAM_BOT_URL", "").strip()
    examples = sorted((ROOT / "examples").glob("*.jpg")) if (ROOT / "examples").exists() else []
    with gr.Blocks(title="Dokter Tani — Halaman Uji") as demo:
        gr.Markdown(
            "# 🌶️ Dokter Tani — Halaman Uji\n"
            "Deteksi penyakit daun cabai dari foto, lalu penanganan berbasis dokumen panduan resmi.\n\n"
            + (f"**Produk utama untuk petani adalah bot Telegram: [{bot_url}]({bot_url})**. " if bot_url else "")
            + "Halaman ini disediakan untuk pengujian.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Foto daun", sources=["upload", "webcam"], height=320)
                btn = gr.Button("Analisa", variant="primary")
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=img_in, label="Contoh foto uji")
            with gr.Column(scale=1):
                probs_out = gr.Label(num_top_classes=3, label="Prediksi model")
                cam_out = gr.Image(label="Grad-CAM: area terang = bagian yang dilihat model", height=260)
        answer = gr.Markdown(label="Jawaban Dokter Tani")
        gr.Markdown("### Tanya lanjutan")
        chat = gr.Chatbot(height=320, label="Percakapan")
        with gr.Row():
            msg = gr.Textbox(placeholder="Contoh: Berapa hari sekali harus disemprot?", show_label=False, scale=5)
            send = gr.Button("Kirim", scale=1)
        gr.Markdown(f"<small>Model: {PREDICTOR.arch} · Knowledge base: {len(KB.chunks)} potongan dokumen · "
                    "Hasil AI adalah bantuan awal, bukan pengganti penyuluh pertanian.</small>")

        btn.click(ui_analyze, [img_in, state], [probs_out, cam_out, answer, chat, state])
        send.click(ui_chat, [msg, chat, state], [chat, msg, state])
        msg.submit(ui_chat, [msg, chat, state], [chat, msg, state])
    return demo


app = gr.mount_gradio_app(api, build_ui(), path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
