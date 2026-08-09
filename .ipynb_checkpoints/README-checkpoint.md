# CS Bot — Decision Tree SLM (Flask + Python)

Chatbot customer service berbasis **Decision Tree** dengan ekstraksi kata kunci otomatis
menggunakan **TF-IDF per paragraf**. Knowledge base disimpan di sisi server dalam format JSON.

---

## Struktur Proyek

```
csbot/
├── app.py              ← Flask application (REST API + routing)
├── nlp_engine.py       ← NLP: tokenisasi, stemming, TF-IDF, decision tree matching
├── requirements.txt
├── templates/
│   └── index.html      ← Single-page UI (Chat · KB Manager · Upload)
└── data/
    ├── knowledge_base.json   ← KB tersimpan {id, keywords, answer, source, created}
    ├── documents.json        ← Metadata dokumen yang diupload
    └── contoh_faq.txt        ← Contoh file upload
```

---

## Instalasi & Menjalankan

```bash
# 1. Buat virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan server
python app.py
# → http://localhost:5000
```

---

## Alur NLP (nlp_engine.py)

```
Teks Input
    │
    ▼
Split per paragraf (baris kosong sebagai pemisah)
    │
    ▼ per paragraf
Tokenisasi  →  Stopword Removal (ID)  →  Stemming (Nazief-Adriani simplified)
    │
    ▼
TF-IDF Scoring antar paragraf
    │
    ▼
Top-N keywords per paragraf  (default N=8)
    │
    ▼
Simpan ke knowledge_base.json
{ "id": "uuid", "keywords": [...], "answer": "teks paragraf", "source": "nama_file.txt" }
```

### Decision Tree Matching

Saat user mengirim pesan:
1. Tokenisasi + stopword removal + stemming query
2. Setiap entry KB di-score:
   ```
   score = coverage×0.45 + precision×0.35 + answer_match×0.20
   ```
3. Entry dengan skor tertinggi dipilih (threshold minimum = 0.08)

---

## REST API

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/api/chat` | Kirim pesan, terima jawaban |
| GET | `/api/kb` | Ambil semua entry KB |
| POST | `/api/kb` | Tambah entry manual |
| PUT | `/api/kb/<id>` | Update entry |
| DELETE | `/api/kb/<id>` | Hapus entry |
| POST | `/api/kb/clear` | Hapus semua / per sumber |
| POST | `/api/upload` | Upload file .txt (multipart atau JSON) |
| POST | `/api/upload/preview` | Preview ekstraksi tanpa menyimpan |
| GET | `/api/documents` | Daftar dokumen terupload |
| DELETE | `/api/documents/<filename>` | Hapus dokumen & entry-nya |
| GET | `/api/stats` | Statistik KB |

### Contoh: Chat
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "jam operasional"}'
```
Response:
```json
{
  "reply": "Layanan pelanggan kami beroperasi setiap hari Senin...",
  "score": 0.662,
  "matched": ["beroperasi", "hari"],
  "source": "contoh_faq.txt",
  "intent": "kb_match"
}
```

### Contoh: Upload File
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@data/contoh_faq.txt"
```

---

## Format File .txt

Pisahkan topik dengan **baris kosong**. Satu paragraf = satu entry KB.

```
Layanan pelanggan kami beroperasi Senin–Jumat 08.00–17.00 WIB.
Sabtu buka setengah hari. Minggu dan libur nasional tutup.

Untuk menghubungi CS, gunakan telepon 0370-123-4567 atau
email cs@example.ac.id atau WhatsApp 0812-3456-7890.

Pendaftaran dapat dilakukan online di website resmi atau
datang langsung ke kantor dengan membawa kartu identitas.
```

---

## Format knowledge_base.json

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "keywords": ["beroperasi", "hari", "senin", "jumat", "pukul"],
    "answer": "Layanan pelanggan kami beroperasi setiap hari Senin sampai Jumat...",
    "source": "contoh_faq.txt",
    "created": "2026-08-08 10:30:00"
  }
]
```

---

## Fitur UI

- **Chat** — percakapan real-time dengan badge kata kunci ter-match & confidence bar
- **Knowledge Base** — CRUD entry, filter per dokumen, search real-time
- **Upload Dokumen** — drag & drop file .txt atau paste teks, preview ekstraksi sebelum simpan
