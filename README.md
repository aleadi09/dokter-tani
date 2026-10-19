# Dokter Tani 🌶️

Bot Telegram yang membantu petani cabai mengenali penyakit daun dari foto, lalu memberi cara penanganan berbasis dokumen panduan resmi dalam bahasa sederhana.

> Final project AI Engineering Bootcamp. Status: **Minggu 2 — model final (EfficientNet-B0), Grad-CAM & upload Hugging Face**.

## Cara kerja (rencana)

```
Foto daun (Telegram) → n8n → Hugging Face Space
                               ├─ CNN classifier + Grad-CAM
                               └─ AI Agent (RAG dokumen panduan)
                       ← diagnosis, gambar Grad-CAM, penanganan, sumber dokumen
```

## Struktur folder

| Folder | Isi |
|---|---|
| `notebooks/` | EDA, training, evaluasi |
| `data/` | Dataset (tidak di-upload ke GitHub, lihat `data/README.md`) |
| `src/model/` | Kode model: training dan inference |
| `src/rag/` | Ingest dokumen panduan, retrieval |
| `src/agent/` | Tools dan prompt agent |
| `app/` | Gradio app untuk Hugging Face Spaces (halaman uji) |
| `n8n/` | Export workflow bot Telegram |
| `docs/` | Dokumen sumber RAG |
| `reports/` | Grafik dan tabel hasil (dibuat otomatis oleh notebook) |
| `CATATAN.md` | Log eksperimen |

## Menjalankan di laptop (VS Code)

1. Buat virtual environment dan install paket:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    | Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Install ekstensi **Python** dan **Jupyter** di VS Code.
3. Taruh dataset di `data/raw/chilli_leaf/` (lihat `data/README.md`).
4. Buka `notebooks/01_eda.ipynb`, pilih kernel `.venv`, lalu **Run All**.

## Menjalankan dengan GPU Colab (untuk training)

Pakai ekstensi **Google Colab** di VS Code (*Select Kernel → Colab*) atau buka notebook di colab.research.google.com. Siapkan dua file di Google Drive, folder `MyDrive/dokter-tani/`:

| File di Drive | Asalnya |
|---|---|
| `chilli_leaf.zip` | **Isi** folder `data/raw/chilli_leaf/` di-zip (bukan folder induknya), supaya path-nya sama dengan `splits.csv` |
| `splits.csv` | Salin dari `data/splits.csv` hasil notebook 01 |

Notebook akan mendeteksi Colab, me-mount Drive, dan mengekstrak zip secara otomatis. Hasil dan model tersimpan di Drive (`reports/` dan `checkpoints/`).

## Dataset

- Utama: [Chilli Leaf Disease Image Dataset (Mendeley Data)](https://data.mendeley.com/datasets/tm3v4zmh7c/1) — 6 kelas, foto lapangan
- Uji generalisasi: [Dataset of chilli and onion plant leaf images (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170091/) — lisensi CC BY-NC

## Hasil

| Model | Parameter | Test accuracy | Test macro F1 | Recall terendah | CPU ms/foto |
|---|---|---|---|---|---|
| Baseline CNN (dari nol) | 1,2 jt | 0,966 | 0,968 | 0,921 | – |
| **EfficientNet-B0 (final)** | 4,0 jt | **0,983** | **0,985** | **0,974** | 27 |

_Link model Hugging Face, demo, dan bot Telegram: akan diisi._

## Disclaimer

Hasil AI adalah bantuan awal, bukan pengganti penyuluh pertanian.
