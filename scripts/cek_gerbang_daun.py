"""Kalibrasi penjaga gerbang daun (CLIP): berapa batas skor yang pas?

1. Taruh 15–30 foto BUKAN daun cabai di  data/bukan_daun/
   (selfie, kucing, makanan, screenshot, buah cabai, bunga, tanah, tangan, dll.)
2. Jalankan dari folder project:   python scripts/cek_gerbang_daun.py
   (pertama kali mengunduh model CLIP ±600 MB)

Hasil: tabel di layar + reports/gerbang_daun.csv. Pilih batas yang meloloskan hampir semua foto daun
dan menolak sebanyak mungkin foto bukan daun, lalu isi Space → Settings → Variables: LEAF_THRESHOLD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model.leaf_gate import LeafGate  # noqa: E402

PER_CLASS = 25
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    gate = LeafGate(threshold=0.5)
    if not gate.enabled:
        raise SystemExit(f"Model CLIP gagal dimuat: {gate.error}")

    rows = []
    splits, data = ROOT / "data" / "splits.csv", ROOT / "data" / "raw" / "chilli_leaf"
    if splits.exists():
        df = pd.read_csv(splits)
        df["rel_path"] = df["rel_path"].str.replace("\\", "/", regex=False)
        test = df[df["split"] == "test"].groupby("label").head(PER_CLASS)
        for _, r in test.iterrows():
            rows.append({"jenis": "daun", "kelas": r["label"], "file": str(data / r["rel_path"])})
    own = ROOT / "data" / "own_photos"                   # foto daun milik sendiri (bila ada)
    rows += [{"jenis": "daun", "kelas": "foto_sendiri", "file": str(p)} for p in sorted(own.rglob("*")) if p.suffix.lower() in EXT]
    neg = ROOT / "data" / "bukan_daun"
    rows += [{"jenis": "bukan daun", "kelas": "bukan_daun", "file": str(p)} for p in sorted(neg.rglob("*")) if p.suffix.lower() in EXT]
    if not any(r["jenis"] == "bukan daun" for r in rows):
        print("⚠️  Folder data/bukan_daun/ kosong: isi dulu dengan foto bukan daun supaya batas bisa diukur.\n")

    for i, r in enumerate(rows, 1):
        with Image.open(r["file"]) as im:
            res = gate.check(im)
        r.update(skor=res["leaf_score"], deskripsi_teratas=res["top"])
        print(f"\r  {i}/{len(rows)}", end="", flush=True)
    print()
    out = pd.DataFrame(rows)
    (ROOT / "reports").mkdir(exist_ok=True)
    out.to_csv(ROOT / "reports" / "gerbang_daun.csv", index=False)

    print("\nSkor daun per kelompok (0 = pasti bukan daun, 1 = pasti daun):")
    print(out.groupby("kelas")["skor"].describe()[["count", "min", "50%", "max"]].round(3).to_string())

    leaf, other = out[out.jenis == "daun"]["skor"], out[out.jenis == "bukan daun"]["skor"]
    print("\nBatas | foto daun lolos | foto bukan-daun ditolak")
    cands = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        lolos = (leaf >= t).mean() if len(leaf) else float("nan")
        tolak = (other < t).mean() if len(other) else float("nan")
        print(f" {t:.1f}  |     {lolos:6.1%}      |     {tolak:6.1%}")
        if len(other) and lolos >= 0.98:
            cands.append((t, tolak))
    if cands:
        top = max(c[1] for c in cands)
        ties = [c for c in cands if c[1] == top]
        mid = (other.max() + leaf.min()) / 2 if other.max() < leaf.min() else 0.5   # tengah celah = ruang aman 2 sisi
        best = min(ties, key=lambda c: abs(c[0] - mid))
        if other.max() < leaf.min():
            print(f"\nCelah: skor bukan-daun tertinggi {other.max():.3f} < skor daun terendah {leaf.min():.3f} → terpisah sempurna")
        print(f"Saran: LEAF_THRESHOLD = {best[0]} (≥98% foto daun lolos, {best[1]:.0%} foto bukan daun ditolak, "
              f"dipilih paling dekat tengah celah)")
    salah = out[(out.jenis == "daun") & (out.skor < 0.5)]
    if len(salah):
        print(f"\nFoto daun dengan skor < 0.5 ({len(salah)}), cek apakah memang sulit dikenali:")
        print(salah[["kelas", "skor", "deskripsi_teratas", "file"]].head(10).to_string(index=False))
    print(f"\nTersimpan: {ROOT / 'reports' / 'gerbang_daun.csv'}")


if __name__ == "__main__":
    main()
