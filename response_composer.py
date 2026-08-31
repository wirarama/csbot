"""
response_composer.py — Ubah jawaban KB (paragraf sumber apa adanya) menjadi
balasan chat yang terasa seperti jawaban natural, tanpa memakai model bahasa
eksternal (tetap 100% lokal/offline, konsisten dengan filosofi decision-tree
csbot).

Pendekatan:
1. Deteksi kategori topik dari kata kunci yang match (waktu, kontak, syarat,
   biaya, lokasi, prosedur, atau umum).
2. Pilih preamble percakapan sesuai kategori (dirotasi acak dari beberapa
   variasi supaya tidak monoton/robotic).
3. Gabungkan preamble + isi jawaban sumber secara gramatikal (tanpa mengubah
   fakta di dalam jawaban itu sendiri — hanya bungkusnya yang berbeda).
4. Tambahkan closer opsional (kadang muncul, kadang tidak) agar terasa seperti
   percakapan Q&A, bukan dump teks.

Hasil compose_reply() dijamin TIDAK PERNAH identik string-nya dengan
`answer` mentah, karena preamble selalu ditambahkan di depan.
"""

import random
import re
from typing import List

# ── Kategori topik → daftar kata kunci pemicu ────────────────────────────────
CATEGORIES = {
    'waktu':    ['jam', 'waktu', 'operasional', 'buka', 'tutup', 'pukul', 'hari', 'libur', 'jadwal'],
    'kontak':   ['telepon', 'email', 'whatsapp', 'wa', 'kontak', 'hubungi', 'nomor', 'telp'],
    'syarat':   ['syarat', 'dokumen', 'persyaratan', 'wajib', 'diperlukan', 'berkas', 'lampiran'],
    'biaya':    ['biaya', 'harga', 'tarif', 'bayar', 'gratis', 'pembayaran', 'ongkos'],
    'lokasi':   ['alamat', 'lokasi', 'kantor', 'gedung', 'tempat', 'ruang'],
    'prosedur': ['daftar', 'pendaftaran', 'cara', 'langkah', 'proses', 'prosedur', 'mengajukan'],
}

# ── Preamble per kategori (beberapa variasi supaya tidak berulang) ──────────
PREAMBLES = {
    'waktu': [
        "Untuk jadwal/jam operasionalnya,",
        "Soal waktu operasional kami,",
        "Baik, terkait jam operasional,",
    ],
    'kontak': [
        "Untuk menghubungi kami,",
        "Soal cara menghubungi kami,",
        "Berikut kontak yang bisa Anda gunakan:",
    ],
    'syarat': [
        "Untuk persyaratannya,",
        "Soal persyaratan yang dibutuhkan,",
        "Berikut syarat yang perlu disiapkan:",
    ],
    'biaya': [
        "Untuk informasi biayanya,",
        "Soal biaya/tarifnya,",
    ],
    'lokasi': [
        "Untuk lokasinya,",
        "Soal alamat kami,",
    ],
    'prosedur': [
        "Untuk prosedurnya,",
        "Soal langkah-langkahnya,",
        "Berikut caranya:",
    ],
    'umum': [
        "Baik, terkait pertanyaan Anda soal {kw},",
        "Mengenai {kw} yang Anda tanyakan,",
        "Soal {kw},",
    ],
}

GENERIC_PREAMBLES = ["Baik, ", "Untuk itu, ", "Oke, ", "Berikut informasinya: "]

# String kosong sengaja diberi bobot lebih besar supaya closer tidak selalu muncul
# (kalau selalu muncul malah terasa template/robotic).
CLOSERS = [
    "", "", "", "",
    " Ada lagi yang ingin ditanyakan? 🙂",
    " Semoga membantu ya!",
    " Kalau masih kurang jelas, silakan tanya lagi.",
]


def _detect_category(matched_kw: List[str]) -> str:
    kws_lower = [k.lower() for k in (matched_kw or [])]
    best_cat, best_hits = None, 0
    for cat, triggers in CATEGORIES.items():
        hits = sum(1 for kw in kws_lower if any(t in kw or kw in t for t in triggers))
        if hits > best_hits:
            best_cat, best_hits = cat, hits
    return best_cat or 'umum'


def _lead_lower(text: str) -> str:
    """Turunkan huruf pertama kata pertama supaya nyambung gramatikal setelah
    preamble yang diakhiri koma. Tidak menyentuh sisa isi (fakta, angka, dll)."""
    if not text:
        return text
    return text[0].lower() + text[1:]


# Kata sambung/generik yang diabaikan saat mengecek tumpang-tindih preamble vs
# awal jawaban sumber (supaya tidak menghasilkan "Untuk menghubungi kami, untuk
# menghubungi CS, ..." yang ganjil).
_CONNECTOR_WORDS = {
    'untuk', 'soal', 'baik', 'terkait', 'mengenai', 'yang', 'anda', 'tanyakan',
    'kami', 'berikut', 'bisa', 'kah', 'nya', 'hal', 'ini', 'oke', 'gunakan',
}


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-zA-Z]+", text.lower())
            if w not in _CONNECTOR_WORDS and len(w) > 2}


def _has_redundant_overlap(preamble: str, answer: str, first_n_words: int = 8) -> bool:
    """True kalau kata inti preamble sudah muncul di awal jawaban sumber
    (mis. preamble 'Untuk menghubungi kami,' vs jawaban yang juga dibuka
    dengan 'Untuk menghubungi CS, ...')."""
    pre_words = _content_words(preamble)
    if not pre_words:
        return False
    ans_words = _content_words(' '.join(answer.split()[:first_n_words]))
    return bool(pre_words & ans_words)


def compose_reply(answer: str, matched_kw: List[str] = None, score: float = 0.0) -> str:
    """
    Susun ulang jawaban KB (paragraf sumber) menjadi balasan chat yang natural:
    preamble kontekstual sesuai topik + isi jawaban (fakta tidak diubah) +
    closer opsional. Selalu menghasilkan string yang berbeda dari `answer` asli.
    """
    answer = (answer or '').strip()
    if not answer:
        return answer

    category = _detect_category(matched_kw)

    if category in PREAMBLES:
        preamble = random.choice(PREAMBLES[category])
        if '{kw}' in preamble:
            top_kw = (matched_kw[0] if matched_kw else 'hal ini')
            preamble = preamble.format(kw=top_kw)
    else:
        preamble = random.choice(GENERIC_PREAMBLES)

    # Kalau preamble & awal jawaban sumber sama-sama membahas frasa yang sama
    # (mis. "Untuk menghubungi kami," + "Untuk menghubungi CS, ..."), jangan
    # digabung koma (jadi dobel) — pisahkan bersih pakai titik dua.
    redundant = _has_redundant_overlap(preamble, answer)

    if redundant or preamble.endswith(':'):
        preamble_clean = preamble.rstrip(':,') + ':'
        body = f"{preamble_clean}\n\n{answer}"
    elif preamble.endswith(' '):
        body = f"{preamble}{answer}"
    else:  # diakhiri koma
        body = f"{preamble} {_lead_lower(answer)}"

    closer = random.choice(CLOSERS)
    return f"{body}{closer}"
