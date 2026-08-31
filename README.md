# CS Bot — Decision Tree SLM (Flask + Python)

Chatbot customer service berbasis **Decision Tree** dengan ekstraksi kata kunci otomatis
menggunakan **TF-IDF per paragraf**. Knowledge base disimpan di sisi server dalam format JSON.

Mendukung **beberapa SLM (Small Language Model) sekaligus** — bisa dibuat kosong, di-*switch*
kapan saja, dan diekspor/diimpor sebagai satu file JSON — serta **web scraping** sebagai
sumber data baru selain upload file `.txt` / paste teks.

---

## Struktur Proyek

```
csbot/
├── app.py              ← Flask application (REST API + routing + manajemen SLM)
├── nlp_engine.py       ← NLP: tokenisasi, stemming, TF-IDF, decision tree matching
├── scraper.py          ← Web scraping: fetch URL + ekstrak teks via HTML tag / CSS class
├── graph_engine.py     ← Proyeksi KB → graph (hierarchy / keyword network)
├── tree_renderer.py    ← Render decision tree ke PNG
├── requirements.txt
├── templates/
│   └── index.html      ← Single-page UI (Chat · KB · Upload · Web Scraping · Tree · Graph)
└── data/
    ├── slm_registry.json     ← Daftar semua SLM + id SLM yang sedang aktif
    ├── slms/
    │   └── <slm_id>.json     ← Isi tiap SLM: {"kb": [...], "documents": [...]}
    ├── knowledge_base.json   ← (peninggalan versi lama — hanya dibaca sekali saat migrasi)
    └── documents.json        ← (peninggalan versi lama — hanya dibaca sekali saat migrasi)
```

### Migrasi otomatis dari versi lama

Sebelumnya semua input (form KB manual + upload dokumen) selalu masuk ke satu
`knowledge_base.json`. Saat `app.py` dijalankan pertama kali setelah update ini, isi
`data/knowledge_base.json` + `data/documents.json` yang lama otomatis dipindahkan menjadi
SLM pertama bernama **"Default"** — data lama tidak hilang. Proses ini hanya berjalan sekali
(dipicu saat `data/slm_registry.json` belum ada).

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

Semua endpoint KB/dokumen/chat/tree/graph di bawah ini beroperasi terhadap **SLM yang
sedang aktif** — lihat bagian [Multi-SLM](#multi-slm-small-language-model) untuk cara
berpindah SLM.

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/api/chat` | Kirim pesan, terima jawaban (dari SLM aktif) |
| GET | `/api/kb` | Ambil semua entry KB (SLM aktif) |
| POST | `/api/kb` | Tambah entry manual (ke SLM aktif) |
| PUT | `/api/kb/<id>` | Update entry |
| DELETE | `/api/kb/<id>` | Hapus entry |
| POST | `/api/kb/clear` | Hapus semua / per sumber |
| POST | `/api/upload` | Upload file .txt (multipart atau JSON) |
| POST | `/api/upload/preview` | Preview ekstraksi tanpa menyimpan |
| POST | `/api/scrape` | Scrape URL (tag/class/selector) → simpan sebagai satu dokumen |
| POST | `/api/scrape/preview` | Preview hasil scraping tanpa menyimpan |
| GET | `/api/documents` | Daftar dokumen (upload & scraping) |
| DELETE | `/api/documents/<filename>` | Hapus dokumen & entry-nya |
| GET | `/api/stats` | Statistik KB (SLM aktif) |
| GET | `/api/slm` | Daftar semua SLM + status aktif |
| POST | `/api/slm` | Buat SLM baru (kosong), `{"name": "..."}` |
| PUT | `/api/slm/<id>` | Ganti nama SLM |
| DELETE | `/api/slm/<id>` | Hapus SLM (tidak bisa hapus SLM terakhir) |
| POST | `/api/slm/<id>/activate` | Switch SLM aktif |
| GET | `/api/slm/<id>/export` | Download satu SLM sebagai file JSON |
| POST | `/api/slm/import` | Import file JSON SLM sebagai SLM baru |

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

## Multi-SLM (Small Language Model)

Setiap **SLM** adalah satu knowledge-base terpisah (kumpulan entry KB + daftar dokumen).
Dropdown "🗂" di header menampilkan SLM yang sedang aktif; tombol **⚙ Kelola SLM** membuka
panel untuk:

- **Buat SLM kosong** — beri nama, langsung aktif, siap diisi lewat tab Upload/Web Scraping/KB manual
- **Switch** — pindah SLM aktif kapan saja lewat dropdown atau tombol "Aktifkan"
- **Rename** / **Hapus** (SLM terakhir tidak bisa dihapus)
- **Export** — download satu SLM (`kb` + `documents`) sebagai satu file `.json`
- **Import** — upload file `.json` hasil export → dibuat sebagai SLM baru (tidak menimpa yang sudah ada)

Semua input — entry manual, upload file/paste teks, maupun web scraping — selalu masuk ke
SLM yang **sedang aktif** saat itu. Jadi untuk memisahkan data per topik/klien: buat SLM
baru → switch ke SLM tersebut → baru upload/scrape/tambah entry.

Format file export/import:
```json
{
  "format": "csbot-slm",
  "version": 1,
  "id": "slm_...",
  "name": "Produk A",
  "exported_at": "2026-08-31 10:00:00",
  "kb": [ { "id": "...", "keywords": [...], "answer": "...", "source": "...", "created": "..." } ],
  "documents": [ { "filename": "...", "paragraphs": 2, "uploaded": "...", "char_count": 120 } ]
}
```

---

## Web Scraping

Tab **🌐 Web Scraping** mengambil teks dari sebuah halaman web tanpa perlu men-download
manual jadi file `.txt`. Isi:

- **URL** halaman
- **HTML tag** (mis. `p`, `div`, `article`) dan/atau **CSS class** (mis. `content`, `faq-item`)
- atau **selector CSS lanjutan** (mis. `#main .faq-item`, `div.article > p`) yang meng-override tag/class

Setiap elemen yang cocok dengan selector diperlakukan sebagai satu paragraf, lalu diproses
lewat pipeline NLP yang sama dengan upload dokumen (tokenisasi → stopword removal → stemming
→ TF-IDF → keyword). **Semua elemen dari satu URL digabung dan disimpan sebagai satu
dokumen sumber** (nama dokumen otomatis dari domain+path URL, atau bisa diisi manual), masuk
ke SLM yang sedang aktif.

```bash
curl -X POST http://localhost:5000/api/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://contoh.com/faq", "tag": "div", "css_class": "faq-item"}'
```

Catatan: fitur ini butuh akses jaringan keluar dari komputer yang menjalankan `app.py`
(package `requests` + `beautifulsoup4`, sudah ada di `requirements.txt`).

---

## Fitur UI

- **Chat** — percakapan real-time dengan badge kata kunci ter-match & confidence bar
- **Knowledge Base** — CRUD entry, filter per dokumen, search real-time
- **Upload Dokumen** — drag & drop file .txt atau paste teks, preview ekstraksi sebelum simpan
- **Web Scraping** — ambil teks dari URL via HTML tag/CSS class, preview sebelum simpan
- **🗂 Kelola SLM** — buat SLM kosong, switch, rename, hapus, export/import JSON
