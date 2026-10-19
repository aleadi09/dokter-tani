"""Siapkan folder Hugging Face Space dari repo ini, lalu (opsional) upload.

Contoh:
  python scripts/build_space.py                                   # hanya membuat folder space_build/
  python scripts/build_space.py --push NAMA_HF/dokter-tani        # buat + upload ke Space
  python scripts/build_space.py --push NAMA_HF/dokter-tani --model-repo NAMA_HF/dokter-tani-cabai

Token Hugging Face (tipe Write) dibaca dari env HF_TOKEN atau ditanyakan.
Secret yang ikut di-set bila ada di .env / environment: GROQ_API_KEY, GEMINI_API_KEY, API_TOKEN.
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK_VERSION = "6.30.0"

SPACE_REQUIREMENTS = f"""--extra-index-url https://download.pytorch.org/whl/cpu
gradio=={SDK_VERSION}
uvicorn
torch
torchvision
numpy
pillow
matplotlib
scikit-learn
sentence-transformers
transformers
safetensors
huggingface_hub
openai
requests
"""

CODE_FILES = ["src/__init__.py", "src/model/__init__.py", "src/model/predictor.py", "src/model/leaf_gate.py", "src/rag/__init__.py",
              "src/rag/knowledge.py", "src/agent/__init__.py", "src/agent/agent.py"]
# Space tipe Docker: aplikasi menjalankan server sendiri (Gradio + API FastAPI di satu port 7860).
# (Space tipe Gradio menjalankan server Gradio sendiri di port 7860 sehingga bentrok dengan API kita.)
DOCKERFILE = """FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \\
    PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
# unduh model embedding saat build supaya Space cepat siap setelah restart
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
# model CLIP untuk penjaga gerbang (foto daun atau bukan)
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
"""

SECRETS = ["GROQ_API_KEY", "GEMINI_API_KEY", "API_TOKEN"]
VARIABLES = ["LLM_PROVIDER", "LLM_MODEL", "TELEGRAM_BOT_URL", "LEAF_THRESHOLD"]


def space_readme(model_repo: str | None) -> str:
    model_line = f"- Model: [{model_repo}](https://huggingface.co/{model_repo})\n" if model_repo else ""
    return f"""---
title: Dokter Tani
emoji: 🌶️
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
pinned: false
startup_duration_timeout: 1h
---

# Dokter Tani — Halaman Uji

Deteksi penyakit daun cabai dari foto (EfficientNet-B0 + Grad-CAM), lalu penanganan berbasis dokumen panduan resmi
(RAG hybrid + agent LLM). Produk utama untuk petani adalah bot Telegram; halaman ini untuk pengujian.

{model_line}- API untuk bot: `/api/health`, `/api/diagnose`, `/api/ask` (lihat docstring di `app.py`)

Hasil AI adalah bantuan awal, bukan pengganti penyuluh pertanian.
"""


def read_env() -> dict:
    vals = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"')
    return {**vals, **{k: v for k, v in os.environ.items() if k in SECRETS + VARIABLES}}


def copy_examples(out: Path, per_class: int = 1):
    """Satu foto test per kelas sebagai contoh di halaman uji (diperkecil agar ringan)."""
    import pandas as pd
    from PIL import Image, ImageOps
    splits, data = ROOT / "data" / "splits.csv", ROOT / "data" / "raw" / "chilli_leaf"
    if not splits.exists() or not data.exists():
        print("  (lewati contoh foto: data/splits.csv atau dataset tidak ada)")
        return
    df = pd.read_csv(splits)
    df["rel_path"] = df["rel_path"].str.replace("\\", "/", regex=False)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    for label, g in df[df["split"] == "test"].groupby("label"):
        for i, rp in enumerate(g["rel_path"].head(per_class)):
            with Image.open(data / rp) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((512, 512))
                im.save(out / "examples" / f"{label}_{i + 1}.jpg", quality=88)


def build(out: Path, model_repo: str | None) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(ROOT / "app" / "app.py", out / "app.py")
    for f in CODE_FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / f
        if src.exists():
            shutil.copy(src, out / f)
        else:
            (out / f).write_text("", encoding="utf-8")
    if not (ROOT / "rag_index" / "meta.json").exists():
        raise SystemExit("rag_index/ belum ada. Jalankan notebook 05 dulu.")
    shutil.copytree(ROOT / "rag_index", out / "rag_index")
    if not model_repo:
        src = ROOT / "export" / "hf_model"
        if not (src / "model.safetensors").exists():
            raise SystemExit("export/hf_model/ belum ada. Jalankan notebook 04 (bagian ekspor), atau pakai --model-repo.")
        (out / "model").mkdir()
        for f in ("model.safetensors", "config.json"):
            shutil.copy(src / f, out / "model" / f)
    copy_examples(out)
    (out / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (out / "README.md").write_text(space_readme(model_repo), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Folder Space siap: {out} ({size:.1f} MB)")
    return out


def push(out: Path, repo_id: str, model_repo: str | None):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN") or getpass.getpass("Token Hugging Face (Write): ").strip()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", space_hardware="cpu-basic", exist_ok=True)
    try:   # pastikan CPU Basic (bukan ZeroGPU), juga untuk Space yang sudah telanjur dibuat
        api.request_space_hardware(repo_id, "cpu-basic")
        print("  hardware: cpu-basic")
    except Exception as e:
        print(f"  (hardware tidak diubah: {str(e)[:150]})")
    env = read_env()
    for k in SECRETS:
        if env.get(k):
            api.add_space_secret(repo_id, k, env[k])
            print(f"  secret {k} di-set")
    for k in VARIABLES:
        if env.get(k):
            api.add_space_variable(repo_id, k, env[k])
    if model_repo:
        api.add_space_variable(repo_id, "MODEL_REPO", model_repo)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="space", commit_message="Deploy Dokter Tani")
    print(f"Selesai. Space: https://huggingface.co/spaces/{repo_id}")
    print(f"API:   https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space/api/health")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "space_build"))
    ap.add_argument("--model-repo", default=None, help="repo model di HF Hub; kosong = bawa folder model/ ke Space")
    ap.add_argument("--push", default=None, metavar="NAMA_HF/NAMA_SPACE")
    a = ap.parse_args()
    folder = build(Path(a.out), a.model_repo)
    if a.push:
        push(folder, a.push, a.model_repo)
