"""Penjaga gerbang: apakah foto ini benar-benar foto daun?

CNN Dokter Tani hanya mengenal 6 kelas daun cabai, sehingga foto apa pun (orang, kucing, layar HP) tetap
dipaksa masuk ke salah satu kelas, sering dengan keyakinan tinggi. Sebelum masuk CNN, foto dicek dulu
dengan CLIP (zero-shot): kemiripan gambar dengan deskripsi teks "foto daun" dibandingkan deskripsi lain.

    gate = LeafGate()
    gate.check(img)  ->  {"is_leaf": True, "leaf_score": 0.93, "top": "a close-up photo of a plant leaf"}

leaf_score = total peluang semua deskripsi "daun". Batas default 0.5 (ubah lewat env LEAF_THRESHOLD).
Bila model CLIP gagal dimuat, gerbang dinonaktifkan (semua foto diteruskan ke CNN) supaya bot tetap jalan.
"""
from __future__ import annotations

import os

import torch
from PIL import Image, ImageOps

MODEL_NAME = os.environ.get("LEAF_GATE_MODEL", "openai/clip-vit-base-patch32")

LEAF_PROMPTS = [
    "a close-up photo of a plant leaf",
    "a photo of a green leaf",
    "a photo of a diseased leaf with spots",
    "a photo of a yellow curled leaf",
    "a photo of a leaf with white powder",
    "a photo of chili plant leaves",
]
OTHER_PROMPTS = [
    "a photo of a person", "a selfie", "a photo of a cat or dog", "a photo of an animal",
    "a photo of food on a plate", "a screenshot of a phone screen", "a document with text",
    "a photo of a room", "a photo of a building or street", "a photo of a car or motorcycle",
    "a photo of red chili peppers", "a photo of a fruit or vegetable", "a photo of a flower",
    "a photo of an everyday object", "a landscape photo", "a cartoon or drawing", "a black image",
]


def _vec(out) -> torch.Tensor:
    """transformers 4.x mengembalikan tensor; 5.x bisa mengembalikan objek output."""
    if isinstance(out, torch.Tensor):
        return out
    for k in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


class LeafGate:
    def __init__(self, model_name: str = MODEL_NAME, threshold: float | None = None, device: str = "cpu"):
        self.threshold = float(threshold if threshold is not None else os.environ.get("LEAF_THRESHOLD", 0.5))
        self.device, self.model_name, self.enabled, self.error = device, model_name, False, None
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
            self.prompts = LEAF_PROMPTS + OTHER_PROMPTS
            with torch.no_grad():
                t = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(device)
                emb = _vec(self.model.get_text_features(**t))
                self.text_emb = emb / emb.norm(dim=-1, keepdim=True)
            self.enabled = True
        except Exception as e:                                  # tanpa internet / model gagal → gerbang mati
            self.error = f"{type(e).__name__}: {e}"

    @torch.no_grad()
    def check(self, img: Image.Image) -> dict:
        if not self.enabled:
            return {"is_leaf": True, "leaf_score": None, "top": None, "gate": "nonaktif"}
        img = ImageOps.exif_transpose(img).convert("RGB")
        x = self.processor(images=img, return_tensors="pt").to(self.device)
        emb = _vec(self.model.get_image_features(**x))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        probs = (self.model.logit_scale.exp() * emb @ self.text_emb.T).softmax(dim=-1)[0]
        leaf_score = float(probs[: len(LEAF_PROMPTS)].sum())
        return {"is_leaf": leaf_score >= self.threshold, "leaf_score": round(leaf_score, 4),
                "top": self.prompts[int(probs.argmax())], "gate": "clip"}
