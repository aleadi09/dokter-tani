# Deploy Dokter Tani ke Hugging Face Spaces

Hasilnya: satu **Space** gratis (CPU) yang berisi halaman uji untuk penilai **dan** API untuk bot Telegram.

```
https://huggingface.co/spaces/NAMA_HF/dokter-tani     ← halaman Space (link demo untuk presentasi)
https://nama-hf-dokter-tani.hf.space/api/health        ← API yang dipanggil n8n / bot
```

## 0. Yang harus sudah ada

| File/folder | Dari | Cek |
|---|---|---|
| `export/hf_model/model.safetensors` + `config.json` | Notebook 04, bagian 7 (ekspor) | Atau model sudah di-upload ke Hub (notebook 04, bagian 8) |
| `rag_index/` | Notebook 05 | Berisi `chunks.jsonl`, `embeddings.npy`, `meta.json` |
| `.env` | Anda | Berisi `GROQ_API_KEY=...` dan `API_TOKEN=...` (bebas, buat sendiri, misalnya 20 karakter acak) |

`API_TOKEN` adalah "kata sandi" API supaya Space Anda tidak dipakai orang lain. Nanti n8n mengirimnya di header `X-API-Key`.

## 1. Coba di laptop dulu

```bash
pip install gradio
python app/app.py
```

Buka http://localhost:7860, upload foto daun, klik **Analisa**, lalu coba tanya lanjutan.
Cek API: http://localhost:7860/api/health

## 2. Buat dan upload Space

Buat token **Write** di huggingface.co → Settings → Access Tokens. Lalu:

```bash
# Model ikut dibawa ke Space (paling sederhana)
python scripts/build_space.py --push NAMA_HF/dokter-tani

# ATAU, bila model sudah di-upload ke Hub di notebook 04:
python scripts/build_space.py --push NAMA_HF/dokter-tani --model-repo NAMA_HF/dokter-tani-cabai
```

Script ini:
1. menyusun folder `space_build/` (app.py, src/, rag_index/, model/, contoh foto, requirements, README Space),
2. membuat Space (SDK Gradio) bila belum ada,
3. menyimpan `GROQ_API_KEY` dan `API_TOKEN` dari `.env` sebagai **Secrets** Space (tidak terlihat publik),
4. meng-upload semuanya.

Token akan ditanyakan lewat kotak input (atau set env `HF_TOKEN`).

## 3. Tunggu build

Buka halaman Space → tab **Logs**. Build pertama sekitar 5–15 menit (install PyTorch CPU + unduh model embedding e5 ±470 MB).
Selesai bila log menampilkan:

```
Siap: efficientnet_b0, 236 chunk, LLM aktif: openai/gpt-oss-120b
```

## 4. Uji API

```bash
curl https://nama-hf-dokter-tani.hf.space/api/health

curl -X POST https://nama-hf-dokter-tani.hf.space/api/diagnose \
  -H "X-API-Key: ISI_API_TOKEN" \
  -F "session_id=uji1" -F "image=@foto_daun.jpg"

curl -X POST https://nama-hf-dokter-tani.hf.space/api/ask \
  -H "X-API-Key: ISI_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"session_id": "uji1", "question": "Berapa hari sekali harus disemprot?"}'
```

Jawaban `/api/diagnose` berisi `text` (jawaban bot), `nama`, `prob`, `top`, dan `gradcam_jpg_base64` (gambar Grad-CAM).
`session_id` = ID chat Telegram, supaya pertanyaan lanjutan nyambung dengan foto terakhir.

## Mengubah sesuatu

| Ingin | Caranya |
|---|---|
| Ganti model LLM | Space → Settings → Variables: `LLM_MODEL` = `openai/gpt-oss-20b` (lebih cepat) |
| Ganti ke Gemini | Variables: `LLM_PROVIDER` = `gemini`, Secrets: `GEMINI_API_KEY` |
| Tampilkan link bot di halaman | Variables: `TELEGRAM_BOT_URL` = `https://t.me/NamaBot` |
| Update kode/dokumen | Ubah di repo, jalankan ulang `build_space.py --push ...` |

## Catatan

- Space gratis **tidur** bila lama tidak dipakai; buka link-nya beberapa menit sebelum presentasi agar sudah bangun.
- Hasil AI adalah bantuan awal, bukan pengganti penyuluh pertanian. Kalimat ini sudah tampil di halaman.
