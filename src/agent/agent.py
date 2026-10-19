"""Dokter Tani — agent yang menyusun jawaban untuk petani dari hasil CNN + dokumen panduan (RAG).

Alur diagnose(): hasil CNN → (keyakinan rendah? minta foto ulang) → cari dokumen → cuaca (opsional)
→ LLM menyusun jawaban → saringan keamanan (merek, takaran & pestisida terlarang) → tambah baris Sumber.

LLM dipanggil lewat API yang kompatibel OpenAI (Groq / Gemini). Tanpa API key atau bila LLM gagal,
agent tetap menjawab dalam "mode tanpa LLM": kalimat paling relevan dikutip langsung dari dokumen.

    agent = DokterTaniAgent(kb, provider="groq")
    agent.diagnose(predictor.predict("daun.jpg"))   # {"text", "mode", "seconds", "tool_calls", "filtered", "sources"}
    agent.ask("Berapa hari sekali harus disemprot?")
"""
from __future__ import annotations

import os
import re
import time

PROVIDERS = {
    "groq": {"env": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1",
             "model": "openai/gpt-oss-120b"},
    "gemini": {"env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "model": "gemini-3.8-flash"},
}

MIN_CONFIDENCE = 0.6          # di bawah ini bot minta foto ulang

# Takaran: "2 ml/liter", "1-2 g per liter", "500 cc per tangki", "dosis 2", "konsentrasi 0,2"
DOSE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*(?:-|–|s\.?d\.?|sampai)\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?:ml|cc|mililiter|g|gr|gram|kg|l|lt|liter)\s*(?:/|per|untuk\s+(?:setiap|tiap))\s*"
    r"(?:\d+\s*)?(?:l|lt|liter|tangki|ha|hektar|hektare|tanaman|pohon|m2|m²|meter)\b"
    r"|\b(?:dosis|konsentrasi|takaran)\s*:?\s*\d",
    re.IGNORECASE,
)

# Bahan aktif yang dilarang di Indonesia (Permentan 43/2019 dan konvensi POPs)
BANNED_PESTICIDES = [
    "endosulfan", "ddt", "aldrin", "dieldrin", "endrin", "heptaklor", "heptachlor", "klordan", "chlordane",
    "lindan", "lindane", "mireks", "mirex", "toksafen", "toxaphene", "paration", "parathion",
    "metamidofos", "methamidophos", "monokrotofos", "monocrotophos", "pentaklorofenol", "dikofol", "dicofol",
]

# Merek dagang pestisida yang sering muncul di dokumen/pertanyaan petani → disamarkan
BRANDS = ["score", "daconil", "antracol", "ridomil", "furadan", "dithane", "curacron", "decis", "confidor",
          "amistartop", "amistar", "nativo", "agrept", "bactocyn", "kocide", "previcur", "cabrio", "polyram"]
_FORM = r"(?:EC|WP|SC|WG|WDG|SL|EW|SP|GR|OD|ZC|CS)"
BRAND_RE = re.compile(rf"\*{{0,2}}\b(?:{'|'.join(BRANDS)})\b(?:\s+gold)?(?:\s+\d{{1,3}}(?:[.,]\d+)?\s?{_FORM})?\b\*{{0,2}}",
                      re.IGNORECASE)
# Pola nama produk + kode formulasi, misalnya "Xyzmax 250 EC", "Abc 80 WP"
FORMULATION_RE = re.compile(rf"\*{{0,2}}\b[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?\s+\d{{1,3}}(?:[.,]\d+)?\s?{_FORM}\b\*{{0,2}}")

# Catatan khusus per kelas yang ditambahkan ke prompt
CLASS_NOTES = {
    "Curl_Virus": ("Catatan: foto hanya menunjukkan gejala daun keriting/kuning; jenis virus TIDAK bisa dipastikan dari foto. "
                   "Jangan menulis nama virus tertentu (misalnya CPSV) di baris Diagnosis. Jelaskan bahwa penyebab paling umum "
                   "adalah virus kuning (Gemini virus) yang ditularkan kutu kebul, dan virus lain yang ditularkan kutu daun juga "
                   "bisa menimbulkan keriting. Utamakan pengendalian serangga penular dan pencabutan tanaman sakit."),
    "Bacterial_Spot": ("Catatan: jangan menyarankan rotasi atau tumpang sari dengan tomat, terung, atau kentang, karena satu "
                       "famili dengan cabai dan bisa menjadi inang penyakit yang sama, walaupun disebut di dokumen."),
}


def pct(prob: float) -> str:
    """Tampilan keyakinan: maksimal 99%, karena model foto tidak pernah benar-benar pasti."""
    return f"{min(float(prob), 0.99):.0%}"


def mask_brands(text: str) -> tuple[str, int]:
    """Ganti merek dagang/nama produk berkode formulasi dengan frasa netral. Kembalikan (teks, jumlah penggantian)."""
    text, n1 = BRAND_RE.subn("produk terdaftar", text)
    text, n2 = FORMULATION_RE.subn("produk terdaftar", text)
    text = re.sub(r"\bmerek\s+(?=produk terdaftar)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(fungisida|insektisida|bakterisida|pestisida|obat)\s+produk terdaftar", r"\1 yang terdaftar",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"\b(produk terdaftar)(?:\s*(?:,|atau|dan)\s*produk terdaftar)+", r"\1", text)   # rapikan pengulangan
    if n1 + n2:
        # sisa kalimat canggung: "(seperti produk "produk terdaftar")", "(misalnya produk terdaftar)", tanda kutip
        text = re.sub(r"\s*\((?:[^()]{0,40}?)[\"“”']?produk terdaftar[\"“”']?\)", "", text)
        text = re.sub(r"[\"“”']produk terdaftar[\"“”']", "produk terdaftar", text)
        text = re.sub(r"\s+([,.;])", r"\1", text)
        text = re.sub(r"\b(produk terdaftar)(?:\s*(?:,|atau|dan)\s*produk terdaftar)+", r"\1", text)
    return text, n1 + n2


# Kata yang menandakan pertanyaan masih seputar tanaman (dipakai di mode tanpa LLM untuk menolak topik lain)
AGRI_WORDS = [
    "cabai", "cabe", "tanam", "daun", "batang", "akar", "buah", "bunga", "bibit", "benih", "semai", "pupuk",
    "hara", "hama", "penyakit", "virus", "jamur", "bakteri", "kutu", "serangga", "semprot", "obat", "pestisida",
    "fungisida", "insektisida", "bercak", "kuning", "keriting", "layu", "busuk", "panen", "kebun", "lahan",
    "sawah", "mulsa", "siram", "air", "hujan", "cuaca", "tanah", "gulma", "kena", "sakit", "sehat", "rontok",
    "penyuluh", "tani", "petani", "makan", "aman",
]

# Kata kunci pencarian dokumen untuk tiap kelas CNN
CLASS_QUERIES = {
    "Bacterial_Spot": "gejala, penyebab, dan pengendalian penyakit bercak bakteri Xanthomonas pada cabai",
    "Cercospora_Leaf_Spot": "gejala, penyebab, dan pengendalian bercak daun cercospora (serkospora) pada cabai",
    "Curl_Virus": "gejala dan pengendalian virus kuning keriting daun cabai yang ditularkan kutu kebul",
    "Powdery_Mildew": "gejala, penyebab, dan pengendalian penyakit embun tepung pada cabai",
    "Nutrition_Deficiency": "gejala kekurangan unsur hara pada daun cabai dan cara pemupukannya",
    "Healthy_Leaf": "cara merawat tanaman cabai agar tetap sehat: pemupukan, pengairan, mulsa, sanitasi",
}

SYSTEM_PROMPT = """Kamu adalah Dokter Tani, asisten untuk petani cabai di Indonesia.
Jawab dalam Bahasa Indonesia yang sederhana, ramah, dan singkat (maksimal ±200 kata), cocok dibaca di Telegram.

Aturan wajib:
1. Gunakan HANYA informasi dari KUTIPAN DOKUMEN yang diberikan. Bila kutipan tidak memuat jawabannya, katakan terus terang bahwa dokumen panduan tidak membahasnya.
2. JANGAN menulis takaran atau dosis (angka ml, gram, cc per liter/tangki/hektar) dan JANGAN menyebut merek dagang, termasuk merek yang ditanyakan petani. Jangan mengulang nama merek itu sama sekali; sebut bahan aktifnya saja (misalnya "fungisida berbahan aktif difenokonazol yang terdaftar"). Boleh menyebut jenis atau bahan aktif dari dokumen, lalu minta petani mengikuti label kemasan dan bertanya ke penyuluh.
3. Jangan pernah menyarankan pestisida terlarang (misalnya endosulfan, DDT, aldrin, dieldrin, klordan, heptaklor, lindan).
4. Selalu tutup dengan saran menghubungi penyuluh pertanian setempat (PPL), terutama bila serangan berat atau meluas.
5. Bila pertanyaan di luar topik tanaman cabai/pertanian, tolak dengan sopan dalam satu-dua kalimat. Jangan menjawab isinya sama sekali, lalu ajak petani bertanya soal tanaman cabai.
6. Jangan menulis daftar sumber; sistem akan menambahkannya sendiri.
7. Diagnosis berasal dari model foto dan bisa keliru; sebutkan tingkat keyakinannya.
8. Tulis teks biasa tanpa format tebal/miring (tanpa tanda ** atau _), karena akan dikirim lewat Telegram."""

DIAGNOSIS_FORMAT = """Susun jawaban dengan format ini:
Diagnosis: <nama kondisi> (keyakinan <persen>)
Gejala: <1-2 kalimat>
Penyebab: <1 kalimat>
Penanganan: <3-4 poin singkat>
Pencegahan: <1-2 poin>
<satu kalimat saran ke penyuluh>"""

HEALTHY_FORMAT = """Susun jawaban dengan format ini:
Diagnosis: Daun Sehat (keyakinan <persen>)
<1 kalimat bahwa tidak terlihat gejala penyakit>
Tips perawatan: <3-4 poin singkat>
<satu kalimat saran ke penyuluh bila nanti muncul gejala>"""


def _sentences(text: str) -> list[str]:
    """Kalimat utuh dari chunk: sambung baris PDF yang terpotong, buang judul markdown dan baris sumber."""
    lines = [l for l in text.splitlines() if not re.match(r"\s*(#|\**sumber\**\s*:)", l, re.I)]
    text = re.sub(r"\n(?!\s*(?:[-•*]|\d+[.)])\s)", " ", "\n".join(lines))
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [" ".join(p.strip(" •-*\t").replace("**", "").split()) for p in parts
            if re.search(r"[.!?)]$", p.strip()) and p.strip(" •-*\t")]


def safety_filter(text: str) -> tuple[str, int]:
    """Buang kalimat yang memuat takaran atau pestisida terlarang. Kembalikan (teks, jumlah kalimat dibuang)."""
    text, removed = mask_brands(text)
    lines = []
    for line in text.splitlines():
        bullet = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)
        prefix = bullet.group(0) if bullet else ""
        body = line[len(prefix):]
        keep = []
        for s in re.split(r"(?<=[.!?])\s+", body):
            low = s.lower()
            if DOSE_RE.search(s) or any(re.search(rf"\b{p}\b", low) for p in BANNED_PESTICIDES):
                removed += 1
            else:
                keep.append(s)
        if keep or not body.strip():
            lines.append(prefix + " ".join(keep))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed


def get_weather(lat: float, lon: float, timeout: float = 8) -> dict | None:
    """Ramalan cuaca 3 hari dari Open-Meteo (gratis, tanpa API key)."""
    import requests
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=timeout, params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
            "current": "temperature_2m,relative_humidity_2m",
            "daily": "precipitation_sum,precipitation_probability_max",
        })
        r.raise_for_status()
        d = r.json()
        return {"suhu": d["current"]["temperature_2m"], "kelembapan": d["current"]["relative_humidity_2m"],
                "hujan_mm": d["daily"]["precipitation_sum"], "peluang_hujan": d["daily"]["precipitation_probability_max"]}
    except Exception:
        return None


def weather_text(w: dict) -> str:
    rain = ", ".join(f"{p}%" for p in w["peluang_hujan"])
    txt = (f"Cuaca saat ini {w['suhu']:.0f}°C, kelembapan {w['kelembapan']:.0f}%. "
           f"Peluang hujan 3 hari ke depan: {rain}.")
    if max(w["peluang_hujan"]) >= 60:
        txt += " Hindari menyemprot menjelang hujan karena obat mudah tercuci; pilih pagi hari yang cerah."
    if w["kelembapan"] >= 85:
        txt += " Kelembapan tinggi memudahkan jamur dan bakteri menyebar, jaga sirkulasi udara di kebun."
    return txt


class DokterTaniAgent:
    def __init__(self, kb, provider: str = "groq", model: str | None = None, api_key: str | None = None,
                 retrieval_mode: str = "hybrid", alpha: float = 0.6, top_k: int = 4, use_llm: bool = True):
        self.kb, self.provider = kb, provider
        self.retrieval_mode, self.alpha, self.top_k = retrieval_mode, alpha, top_k
        cfg = PROVIDERS[provider]
        self.model = model or cfg["model"]
        key = api_key or os.environ.get(cfg["env"])
        self.client = None
        if use_llm and key:
            from openai import OpenAI
            self.client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=60, max_retries=2)
        self.reset()

    @property
    def llm_enabled(self) -> bool:
        return self.client is not None

    def reset(self):
        self.diag = None
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_error = None

    # ------------------------------------------------------------ tools
    def _retrieve(self, query: str, disease: str | None) -> list[dict]:
        return self.kb.search(query, k=self.top_k, disease=disease, mode=self.retrieval_mode, alpha=self.alpha)

    @staticmethod
    def _context(chunks: list[dict]) -> str:
        from src.rag.knowledge import KnowledgeBase
        return "\n\n".join(f"[{i}] {KnowledgeBase.cite(c)}\n{c['text']}" for i, c in enumerate(chunks, 1))

    @staticmethod
    def _sources_line(chunks: list[dict]) -> str:
        from src.rag.knowledge import KnowledgeBase
        cites = list(dict.fromkeys(KnowledgeBase.cite(c) for c in chunks))
        return "Sumber: " + "; ".join(cites[:3])

    # ------------------------------------------------------------ LLM / tanpa LLM
    def _llm(self, user_msg: str) -> str | None:
        msgs = [self.messages[0]] + self.messages[1:][-8:] + [{"role": "user", "content": user_msg}]
        try:
            extra = {"reasoning_effort": "low"} if "gpt-oss" in self.model else {}   # model penalar: jangan habiskan token
            out = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.2,
                                                      max_tokens=1500, **extra)
            return out.choices[0].message.content.strip()
        except Exception as e:                      # kuota habis, jaringan, model salah → pakai mode tanpa LLM
            self.last_error = f"{type(e).__name__}: {e}"
            return None

    def _extractive(self, query: str, chunks: list[dict], disease: str | None, n: int = 4) -> list[str]:
        """Pilih kalimat dokumen yang paling banyak berbagi kata dengan pertanyaan / kelas penyakit."""
        from src.rag.knowledge import DISEASE_KEYWORDS
        words = set(re.findall(r"\w{4,}", query.lower()))
        kws = DISEASE_KEYWORDS.get(disease, []) if disease else []
        cand = []
        for rank, c in enumerate(chunks):
            for s in _sentences(c["text"]):
                low = s.lower()
                if not 40 <= len(s) <= 350 or sum(ch.isdigit() for ch in s) > 0.15 * len(s):
                    continue
                score = len(words & set(re.findall(r"\w{4,}", low))) + 2 * sum(k in low for k in kws)
                if score:
                    cand.append((score - 0.5 * rank, s))
        picked = []
        for _, s in sorted(cand, key=lambda x: -x[0]):
            s, _ = safety_filter(s)
            if s and s not in picked:
                picked.append(s)
            if len(picked) == n:
                break
        return picked

    def _finish(self, text: str, chunks: list[dict], t0: float, tools: list[str], mode: str,
                diagnosis: bool = False, header: str | None = None) -> dict:
        text = re.sub(r"(?im)^\s*\**sumber\**\s*:.*$", "", text).strip()       # sumber selalu dari sistem
        text, filtered = safety_filter(text)
        text = text.replace("**", "")                                          # Telegram: tanpa format tebal Markdown
        text = re.sub(r"(keyakinan\s*:?\s*)100\s*%", r"\g<1>99%", text, flags=re.IGNORECASE)   # tidak pernah 100%
        if diagnosis and "diagnosis:" not in text.lower() and header:
            text = f"{header}\n\n{text}"
        if "penyuluh" not in text.lower():
            text += "\n\nUntuk kepastian dan takaran obat, konsultasikan dengan penyuluh pertanian (PPL) setempat."
        if chunks:
            text += "\n\n" + self._sources_line(chunks)
        return {"text": text, "mode": mode, "seconds": round(time.time() - t0, 1), "tool_calls": tools,
                "filtered": filtered, "sources": chunks}

    def _answer(self, prompt: str, query: str, chunks: list[dict], disease: str | None, fallback_intro: str) -> tuple[str, str]:
        if self.llm_enabled:
            text = self._llm(prompt)
            if text:
                return text, f"LLM {self.model}"
        lines = self._extractive(query, chunks, disease)
        body = "\n".join(f"• {s}" for s in lines) if lines else "• Dokumen panduan belum membahas hal ini secara rinci."
        mode = "tanpa LLM" if not self.llm_enabled else "tanpa LLM (LLM gagal)"
        return f"{fallback_intro}\n{body}", mode

    # ------------------------------------------------------------ API publik
    def diagnose(self, diag: dict, location: tuple[float, float] | None = None, question: str | None = None) -> dict:
        """Mulai percakapan baru dari hasil CNN (dict dari DokterTaniPredictor.predict)."""
        from src.model.predictor import display_name
        t0 = time.time()
        self.reset()
        label, prob = diag["label"], float(diag["prob"])
        nama = diag.get("nama") or display_name(label)
        header = f"Diagnosis: {nama} (keyakinan {pct(prob)})"

        if prob < MIN_CONFIDENCE:
            alt = diag.get("top3") or diag.get("top") or []
            alt = [a for a in alt if a.get("label") != label][:2]          # jangan ulang kelas utama
            alt_txt = ", ".join(f"{a['nama']} {pct(a['prob'])}" for a in alt)
            text = (f"Foto belum cukup jelas: model hanya {pct(prob)} yakin ({nama}"
                    + (f"; kemungkinan lain: {alt_txt}" if alt_txt else "") + ").\n\n"
                    "Mohon kirim foto ulang dengan tips berikut:\n"
                    "• satu daun yang bergejala, memenuhi sebagian besar foto\n"
                    "• cahaya terang (siang hari, tidak membelakangi matahari), tidak buram\n"
                    "• latar polos bila memungkinkan, misalnya kertas putih\n\n"
                    "Bila foto ulang tetap ragu, tunjukkan tanaman langsung ke penyuluh pertanian setempat.")
            self.diag = diag
            return {"text": text, "mode": "aturan", "seconds": round(time.time() - t0, 1), "tool_calls": [],
                    "filtered": 0, "sources": []}

        self.diag = diag
        tools = ["cari_dokumen"]
        query = CLASS_QUERIES.get(label, f"gejala dan pengendalian {nama} pada cabai")
        if question:
            query = f"{question} {query}"
        chunks = self._retrieve(query, label)

        weather = None
        if location:
            tools.append("cuaca")
            weather = get_weather(*location)

        fmt = HEALTHY_FORMAT if label == "Healthy_Leaf" else DIAGNOSIS_FORMAT
        prompt = (f"Hasil foto: {nama} (label {label}), keyakinan {pct(prob)}.\n"
                  + (f"Pertanyaan petani: {question}\n" if question else "")
                  + (f"Cuaca: {weather_text(weather)}\n" if weather else "")
                  + f"\nKUTIPAN DOKUMEN:\n{self._context(chunks)}\n\n{fmt}"
                  + (f"\n\n{CLASS_NOTES[label]}" if label in CLASS_NOTES else ""))
        intro = header + ("\n\nTips perawatan dari dokumen panduan:" if label == "Healthy_Leaf"
                          else "\n\nMenurut dokumen panduan:")
        text, mode = self._answer(prompt, query, chunks, label, intro)
        if weather and not self.llm_enabled:
            text += "\n\n" + weather_text(weather)

        self.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
        return self._finish(text, chunks, t0, tools, mode, diagnosis=True, header=header)

    def ask(self, question: str, location: tuple[float, float] | None = None) -> dict:
        """Pertanyaan lanjutan; agent ingat diagnosis dan jawaban sebelumnya."""
        t0 = time.time()
        off_topic = not any(w in question.lower() for w in AGRI_WORDS)
        if off_topic:   # tidak ada satu pun kata pertanian → tolak langsung dengan aturan (cepat, tanpa LLM)
            text = ("Maaf, saya hanya bisa membantu soal tanaman cabai: penyakit daun, perawatan, dan "
                    "pemupukan. Silakan kirim foto daun cabai atau ceritakan masalah tanaman Anda.")
            return {"text": text, "mode": "aturan", "seconds": round(time.time() - t0, 1), "tool_calls": [],
                    "filtered": 0, "sources": []}

        label = self.diag["label"] if self.diag else None
        nama = (self.diag.get("nama") if self.diag else None) or ""
        tools = ["cari_dokumen"]
        chunks = self._retrieve(f"{question} {nama}".strip(), label)
        weather = None
        if location:
            tools.append("cuaca")
            weather = get_weather(*location)

        prompt = (f"Pertanyaan lanjutan petani: {question}\n"
                  + (f"(Diagnosis sebelumnya: {nama})\n" if nama else "")
                  + (f"Cuaca: {weather_text(weather)}\n" if weather else "")
                  + f"\nKUTIPAN DOKUMEN:\n{self._context(chunks)}\n\n"
                    "Jawab langsung pertanyaannya dalam 2-5 kalimat atau poin singkat."
                  + (f"\n\n{CLASS_NOTES[label]}" if label in CLASS_NOTES else ""))
        text, mode = self._answer(prompt, question, chunks, label, "Menurut dokumen panduan:")
        self.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
        return self._finish(text, chunks, t0, tools, mode)
