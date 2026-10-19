"""Dokter Tani — knowledge base RAG dari dokumen panduan penyakit cabai.

Alur: baca PDF/MD/TXT → bersihkan watermark/header berulang → potong (chunk) → beri tag penyakit
→ embedding → simpan ke folder (chunks.jsonl + embeddings.npy) → cari dengan cosine similarity.

Vector store sengaja dibuat sederhana (numpy): ratusan chunk cukup dicari dalam milidetik,
tanpa server database, dan mudah dibawa ke Hugging Face Spaces.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# Kata kunci untuk menandai chunk dengan kelas model (nama kelas = nama folder dataset)
DISEASE_KEYWORDS = {
    "Bacterial_Spot": ["bercak bakteri", "xanthomonas", "bercak kering bakteri", "bacterial spot"],
    "Cercospora_Leaf_Spot": ["cercospora", "serkospora", "mata katak", "frog eye", "frogeye"],
    "Curl_Virus": ["virus kuning", "keriting", "gemini", "begomovirus", "kutu kebul", "bemisia", "leaf curl"],
    "Powdery_Mildew": ["embun tepung", "leveillula", "powdery mildew"],
    "Nutrition_Deficiency": ["unsur hara", "defisiensi", "kekurangan hara", "kekurangan unsur", "klorosis",
                             "nitrogen", "kalium", "magnesium", "fosfor", "kalsium", "pengapuran"],
    "Healthy_Leaf": ["pemupukan", "pupuk dasar", "pupuk susulan", "mulsa", "persemaian", "penyiangan",
                     "jarak tanam", "pengairan", "budidaya", "daun sehat"],
}

# Katalog dokumen yang diketahui → judul rapi untuk sitasi. Dicocokkan dari kata di nama file.
KNOWN_SOURCES = [
    (["sni", "babel", "brmp", "bsip"], "Penerapan SNI 4480:2016 sebagai Pedoman Pengendalian Penyakit pada Tanaman Cabai", "BSIP/BRMP Bangka Belitung", "2024"),
    (["jambi", "14book", "hama_dan_penyakit", "hama-dan-penyakit"], "Hama dan Penyakit pada Tanaman Cabai serta Pengendaliannya", "BPTP Jambi", "2014"),
    (["kuning", "sumut", "sebayang"], "Teknik Pengendalian Penyakit Kuning pada Tanaman Cabai", "BPTP Sumatera Utara", "2013"),
    (["sop"], "SOP Budidaya Cabai Merah", "Puslitbang Hortikultura", "2014"),
    (["buleleng", "serkospora", "cercospora"], "Penyakit Bercak Daun Serkospora pada Tanaman Cabai", "Distankan Buleleng", "2025"),
    (["ringkasan"], "Ringkasan Dokter Tani (disusun dari sumber resmi)", "Tim Dokter Tani", "2026"),
    (["balitsa", "penyakit_penting", "penyakit-penting", "m-31"], "Penyakit Penting pada Tanaman Cabai dan Pengendaliannya", "Balitsa", "2007"),
    (["budidaya", "ptt"], "Budidaya Tanaman Cabai Merah", "Balitsa", "2005"),
]


def source_info(filename: str) -> dict:
    low = filename.lower()
    for keys, title, pub, year in KNOWN_SOURCES:
        if any(k in low for k in keys):
            return {"judul": title, "penerbit": pub, "tahun": year}
    return {"judul": Path(filename).stem.replace("_", " "), "penerbit": "", "tahun": ""}


def tag_diseases(text: str) -> list[str]:
    low = text.lower()
    return [c for c, kws in DISEASE_KEYWORDS.items() if any(k in low for k in kws)]


# ---------------------------------------------------------------- membaca dokumen
def _read_pdf(path: Path) -> list[dict]:
    import pymupdf
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append({"text": page.get_text("text"), "page": i})
    return pages


def _strip_repeated_lines(pages: list[dict], min_share: float = 0.5) -> list[dict]:
    """Buang baris yang muncul di banyak halaman (watermark, header, footer)."""
    if len(pages) < 4:
        return pages
    norm = lambda s: re.sub(r"\d+", "#", s.strip().lower())
    counts = Counter()
    for p in pages:
        counts.update({norm(l) for l in p["text"].splitlines() if l.strip()})
    limit = max(3, int(len(pages) * min_share))
    repeated = {l for l, n in counts.items() if n >= limit}
    return [{**p, "text": "\n".join(l for l in p["text"].splitlines() if norm(l) not in repeated)} for p in pages]


def _clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # sambung kata terpotong di akhir baris
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(docs_dir: str | Path) -> list[dict]:
    """Kembalikan daftar halaman: {text, source, page}. Markdown/TXT dianggap satu halaman per bagian."""
    docs_dir = Path(docs_dir)
    out = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            pages = _strip_repeated_lines(_read_pdf(path))
        elif path.suffix.lower() in {".md", ".txt"}:
            raw = path.read_text(encoding="utf-8")
            parts = re.split(r"\n(?=## )", raw)                # satu bagian per heading level 2
            pages = [{"text": p, "page": None} for p in parts]
        else:
            continue
        for p in pages:
            t = _clean(p["text"])
            if len(t) > 40:
                out.append({"text": t, "source": path.name, "page": p["page"]})
    return out


def chunk_documents(pages: list[dict], size: int = 900, overlap: int = 150) -> list[dict]:
    """Potong per paragraf sampai ±size karakter, dengan overlap agar konteks tidak terputus."""
    chunks = []
    for p in pages:
        paras = [x.strip() for x in re.split(r"\n\s*\n", p["text"]) if x.strip()]
        buf = ""
        for para in paras:
            if len(buf) + len(para) + 1 <= size or not buf:
                buf = f"{buf}\n{para}".strip()
            else:
                chunks.append({**p, "text": buf})
                buf = (buf[-overlap:] + "\n" + para).strip()
            while len(buf) > size * 1.5:                       # paragraf sangat panjang
                chunks.append({**p, "text": buf[:size]})
                buf = buf[size - overlap:]
        if buf:
            chunks.append({**p, "text": buf})
    for i, c in enumerate(chunks):
        c["id"] = i
        c["tags"] = tag_diseases(c["text"])
        c.update(source_info(c["source"]))
    return chunks


# ---------------------------------------------------------------- embedding
class E5Embedder:
    """intfloat/multilingual-e5-small: model embedding multibahasa kecil (±118 jt parameter), paham Bahasa Indonesia."""
    name = "intfloat/multilingual-e5-small"

    def __init__(self, name: str | None = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.name = name or self.name
        self.model = SentenceTransformer(self.name, device=device)

    def encode_passages(self, texts):
        return self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                                 batch_size=32, show_progress_bar=True)

    def encode_query(self, text):
        return self.model.encode([f"query: {text}"], normalize_embeddings=True)[0]


class TfidfEmbedder:
    """Pembanding sederhana (berbasis kata), tanpa model neural."""
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)

    def encode_passages(self, texts):
        m = self.vec.fit_transform(texts)
        return self._norm(m.toarray())

    def encode_query(self, text):
        return self._norm(self.vec.transform([text]).toarray())[0]

    @staticmethod
    def _norm(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------- knowledge base
class KnowledgeBase:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray, embedder):
        self.chunks, self.emb, self.embedder = chunks, embeddings.astype(np.float32), embedder
        self._lex = None   # index TF-IDF untuk mode lexical/hybrid, dibuat saat pertama dipakai

    @classmethod
    def build(cls, docs_dir, embedder, size: int = 900, overlap: int = 150):
        chunks = chunk_documents(load_documents(docs_dir), size=size, overlap=overlap)
        if not chunks:
            raise ValueError(f"Tidak ada teks yang terbaca di {docs_dir}")
        return cls(chunks, np.asarray(embedder.encode_passages([c["text"] for c in chunks])), embedder)

    def save(self, out_dir):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.emb)
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (out_dir / "meta.json").write_text(json.dumps({"embedder": self.embedder.name, "n_chunks": len(self.chunks)},
                                                      indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir, embedder=None):
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        if embedder is None and meta["embedder"] == TfidfEmbedder.name:
            embedder = TfidfEmbedder()
            embedder.encode_passages([c["text"] for c in chunks])   # TF-IDF perlu di-fit ulang pada chunk
        elif embedder is None:
            embedder = E5Embedder(meta["embedder"])
        return cls(chunks, np.load(index_dir / "embeddings.npy"), embedder)

    def _lexical_scores(self, query: str) -> np.ndarray:
        if self._lex is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)   # baris sudah dinormalisasi L2
            self._lex = (vec, vec.fit_transform([c["text"] for c in self.chunks]))
        vec, mat = self._lex
        return np.asarray((mat @ vec.transform([query]).T).toarray()).ravel().astype(np.float32)

    def search(self, query: str, k: int = 4, disease: str | None = None, boost: float = 0.08,
               mode: str = "dense", alpha: float = 0.6) -> list[dict]:
        """Cari k chunk paling relevan. `disease` = hasil klasifikasi model → chunk bertag kelas itu diberi bonus skor.

        mode: "dense" (embedding index), "lexical" (TF-IDF, pencocokan kata), atau "hybrid"
        (alpha * dense + (1 - alpha) * lexical, masing-masing dinormalisasi 0..1 per pertanyaan).
        """
        bonus = boost * np.array([disease in c["tags"] for c in self.chunks], dtype=np.float32) if disease else 0.0
        if mode in ("dense", "hybrid"):
            dense = self.emb @ np.asarray(self.embedder.encode_query(query), dtype=np.float32) + bonus
        if mode in ("lexical", "hybrid"):
            lexical = self._lexical_scores(query) + bonus
        if mode == "dense":
            scores = dense
        elif mode == "lexical":
            scores = lexical
        elif mode == "hybrid":
            mm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
            scores = alpha * mm(dense) + (1 - alpha) * mm(lexical)
        else:
            raise ValueError(f"mode tidak dikenal: {mode}")
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    @staticmethod
    def cite(chunk: dict) -> str:
        bits = [chunk["judul"]]
        meta = ", ".join(x for x in (chunk.get("penerbit"), chunk.get("tahun")) if x)
        if meta:
            bits.append(f"({meta})")
        page = chunk.get("page")
        try:
            page = int(page) if page is not None and page == page else None   # page == page → bukan NaN
        except (TypeError, ValueError):
            page = None
        if page:
            bits.append(f"hal. {page}")
        return " ".join(bits)
