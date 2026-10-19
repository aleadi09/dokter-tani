"""Dokter Tani — prediksi penyakit daun cabai + Grad-CAM.

Satu modul yang dipakai notebook, aplikasi Gradio, bot, dan ikut di-upload ke Hugging Face.
Hanya butuh: torch, torchvision, numpy, pillow (+ safetensors / huggingface_hub untuk format Hub).

    p = DokterTaniPredictor("checkpoints/dokter_tani_final.pt")   # atau folder ekspor / from_hub(...)
    p.predict("daun.jpg")          # {"label", "nama", "prob", "top"}
    overlay, cam = p.gradcam("daun.jpg")
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

DISPLAY_NAMES = {
    "Bacterial_Spot": "Bercak Bakteri",
    "Cercospora_Leaf_Spot": "Bercak Daun Cercospora",
    "Curl_Virus": "Virus Keriting Daun",
    "Healthy_Leaf": "Daun Sehat",
    "Nutrition_Deficiency": "Kekurangan Nutrisi",
    "Powdery_Mildew": "Embun Tepung",
}


def display_name(label):
    """Nama kelas dalam Bahasa Indonesia (fallback: label dengan spasi)."""
    return DISPLAY_NAMES.get(label, str(label).replace("_", " "))


def load_image(img):
    """Path / PIL -> PIL RGB dengan orientasi EXIF sudah dibetulkan (foto HP sering berputar)."""
    if not isinstance(img, Image.Image):
        with Image.open(img) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    return ImageOps.exif_transpose(img).convert("RGB")


def leaf_mask(img, min_sat=0.18, min_val=0.12):
    """Perkiraan kasar area daun: piksel berwarna (saturasi cukup), bukan latar putih/abu-abu/hitam.

    Mengembalikan array float32 HxW bernilai 0/1, seukuran gambar.
    """
    hsv = np.asarray(load_image(img).convert("HSV"), dtype=np.float32) / 255.0
    mask = (hsv[..., 1] > min_sat) & (hsv[..., 2] > min_val)
    return mask.astype(np.float32)


def _build_model(arch, num_classes):
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        target = m.layer4
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
        target = m.features[-1]
    else:
        raise ValueError(f"Arsitektur tidak dikenal: {arch}")
    return m, target


def _colormap(x):
    """Peta warna ala 'jet' tanpa matplotlib: x HxW (0..1) -> HxWx3 uint8."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


class DokterTaniPredictor:
    def __init__(self, path, device=None):
        path = Path(path)
        if path.is_dir():   # folder ekspor Hugging Face: model.safetensors + config.json
            from safetensors.torch import load_file
            cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
            state = load_file(str(path / "model.safetensors"))
        else:               # checkpoint .pt dari notebook 03
            ck = torch.load(path, map_location="cpu", weights_only=False)
            cfg = {k: ck[k] for k in ("arch", "classes", "img_size", "mean", "std")}
            state = ck["model"]

        self.cfg = cfg
        self.arch = cfg["arch"]
        self.classes = list(cfg["classes"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self._target = _build_model(self.arch, len(self.classes))
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.tfm = transforms.Compose([
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])

    @classmethod
    def from_hub(cls, repo_id, token=None, device=None):
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo_id, token=token), device=device)

    def _tensor(self, img):
        return self.tfm(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, img, top_k=3):
        probs = torch.softmax(self.model(self._tensor(load_image(img))), 1)[0].cpu()
        top = probs.topk(min(top_k, len(self.classes)))
        top3 = [{"label": self.classes[i], "nama": display_name(self.classes[i]), "prob": float(p)}
                for p, i in zip(top.values.tolist(), top.indices.tolist())]
        return {**top3[0], "top": top3}

    def gradcam(self, img, class_idx=None, alpha=0.45):
        """Grad-CAM pada lapisan konvolusi terakhir.

        Mengembalikan (overlay uint8 HxWx3, cam float32 HxW 0..1), seukuran foto asli.
        Default: kelas yang diprediksi model.
        """
        pil = load_image(img)
        acts = {}

        def hook(_, __, out):
            acts["a"] = out
            out.register_hook(lambda g: acts.__setitem__("g", g))

        h = self._target.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                logits = self.model(self._tensor(pil))
                idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
                self.model.zero_grad(set_to_none=True)
                logits[0, idx].backward()
        finally:
            h.remove()

        a, g = acts["a"][0].detach(), acts["g"][0].detach()   # C x h x w
        cam = torch.relu((g.mean((1, 2), keepdim=True) * a).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = np.asarray(Image.fromarray(cam.cpu().numpy().astype(np.float32))
                         .resize(pil.size, Image.BILINEAR), dtype=np.float32).clip(0, 1)

        base = np.asarray(pil, dtype=np.float32)
        overlay = ((1 - alpha) * base + alpha * _colormap(cam)).clip(0, 255).astype(np.uint8)
        return overlay, cam
