"""
app.py — Flask CS Chatbot with Decision Tree SLM (multi-SLM edition)

Sebelumnya semua input (form KB manual + upload file/paste teks) selalu
masuk ke satu knowledge_base.json yang sama. Sekarang bisa punya beberapa
"SLM" (Small Language Model / knowledge-base container) sekaligus:

  - Buat SLM baru yang kosong
  - Switch (aktifkan) SLM mana yang sedang dipakai
  - Semua input (KB manual, upload dokumen, web scraping) masuk ke SLM
    yang sedang aktif
  - Export / import satu SLM sebagai satu file JSON

Data lama (data/knowledge_base.json + data/documents.json) otomatis
dimigrasikan menjadi SLM pertama bernama "Default" saat pertama kali
dijalankan setelah update ini.
"""

import os, json, uuid, datetime, re
from pathlib import Path
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, abort, Response, send_file
import io

import requests as _requests

from nlp_engine import parse_text_to_entries, query_kb, extract_keywords_tfidf, tokenize
from tree_renderer import build_tree_png
from graph_engine import build_graph, build_keyword_network, graph_stats, kb_from_graph
from scraper import scrape as scrape_url
from response_composer import compose_reply

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Lokasi data lama (single-KB, sebelum fitur multi-SLM)
KB_FILE   = DATA_DIR / 'knowledge_base.json'
DOCS_FILE = DATA_DIR / 'documents.json'

# Lokasi data baru (multi-SLM)
SLM_DIR        = DATA_DIR / 'slms'
REGISTRY_FILE  = DATA_DIR / 'slm_registry.json'
SLM_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text('utf-8'))
        except Exception:
            pass
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')

def now_str() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ══════════════════════════════════════════════════════════════════════════════
# SLM REGISTRY — multi knowledge-base container
# ══════════════════════════════════════════════════════════════════════════════
def _new_slm_id() -> str:
    return 'slm_' + uuid.uuid4().hex[:10]

def _slm_file(slm_id: str) -> Path:
    return SLM_DIR / f'{slm_id}.json'

def _bootstrap_registry() -> dict:
    """
    Dipanggil sekali saat slm_registry.json belum ada.
    Migrasikan data/knowledge_base.json + data/documents.json (kalau ada)
    menjadi SLM pertama bernama "Default", supaya data lama tidak hilang.
    """
    slm_id = _new_slm_id()
    legacy_kb   = load_json(KB_FILE, [])
    legacy_docs = load_json(DOCS_FILE, [])
    save_json(_slm_file(slm_id), {'kb': legacy_kb, 'documents': legacy_docs})

    reg = {
        'active': slm_id,
        'slms': [{'id': slm_id, 'name': 'Default', 'created': now_str()}],
    }
    save_json(REGISTRY_FILE, reg)
    return reg

def load_registry() -> dict:
    reg = load_json(REGISTRY_FILE, None)
    if reg is None:
        reg = _bootstrap_registry()
    reg.setdefault('slms', [])
    return reg

def save_registry(reg: dict):
    save_json(REGISTRY_FILE, reg)

def get_active_id():
    reg = load_registry()
    ids = {s['id'] for s in reg['slms']}
    active = reg.get('active')
    if active not in ids:
        active = reg['slms'][0]['id'] if reg['slms'] else None
    return active

def load_slm_data(slm_id: str) -> dict:
    if not slm_id:
        return {'kb': [], 'documents': []}
    return load_json(_slm_file(slm_id), {'kb': [], 'documents': []})

def save_slm_data(slm_id: str, data: dict):
    if not slm_id:
        return
    save_json(_slm_file(slm_id), data)


# ── KB / Documents accessors — sekarang mengarah ke SLM yang sedang aktif ──────
def load_kb() -> list:
    return load_slm_data(get_active_id()).get('kb', [])

def save_kb(kb: list):
    sid = get_active_id()
    data = load_slm_data(sid)
    data['kb'] = kb
    save_slm_data(sid, data)

def load_docs() -> list:
    return load_slm_data(get_active_id()).get('documents', [])

def save_docs(docs: list):
    sid = get_active_id()
    data = load_slm_data(sid)
    data['documents'] = docs
    save_slm_data(sid, data)


# ── Intent detection ──────────────────────────────────────────────────────────
GREETINGS = ['halo','hai','hi','hello','selamat','pagi','siang','sore','malam','hei','assalamualaikum']
THANKS    = ['terima kasih','makasih','thanks','thank you','thx']
FAREWELLS = ['bye','sampai jumpa','dadah','selamat tinggal']

def detect_intent(text: str):
    t = text.lower()
    if any(g in t for g in GREETINGS): return 'greet'
    if any(g in t for g in THANKS):    return 'thanks'
    if any(g in t for g in FAREWELLS): return 'bye'
    return None

def greeting_text() -> str:
    h = datetime.datetime.now().hour
    part = 'Pagi' if h < 11 else ('Siang' if h < 15 else ('Sore' if h < 18 else 'Malam'))
    return f'Selamat {part}! 👋 Saya CS Bot siap membantu Anda. Ada yang bisa saya bantu?'

FALLBACKS = [
    'Maaf, saya belum memiliki informasi mengenai hal tersebut. Silakan hubungi kami langsung.',
    'Pertanyaan Anda belum tercakup dalam knowledge base saya. Coba gunakan kata kunci yang berbeda.',
    'Saya tidak menemukan jawaban yang sesuai. Tim CS kami siap membantu di jam kerja 08.00–17.00 WIB.',
]
_fb_idx = 0
def fallback_text() -> str:
    global _fb_idx
    t = FALLBACKS[_fb_idx % len(FALLBACKS)]
    _fb_idx += 1
    return t

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES — Pages
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')

# ══════════════════════════════════════════════════════════════════════════════
# API — Chat
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/chat', methods=['POST'])
def api_chat():
    body = request.get_json(force=True)
    text = (body.get('message') or '').strip()
    if not text:
        return jsonify(error='Empty message'), 400

    intent = detect_intent(text)
    if intent == 'greet':
        return jsonify(reply=greeting_text(), score=1.0, matched=[], source=None, intent='greet')
    if intent == 'thanks':
        return jsonify(reply='Sama-sama! 😊 Ada pertanyaan lain yang bisa saya bantu?',
                       score=1.0, matched=[], source=None, intent='thanks')
    if intent == 'bye':
        return jsonify(reply='Terima kasih telah menghubungi kami. Sampai jumpa! 👋',
                       score=1.0, matched=[], source=None, intent='bye')

    kb = load_kb()
    result = query_kb(kb, text)
    if result:
        composed = compose_reply(
            result['entry']['answer'],
            result['matched_kw'],
            result['score'],
        )
        return jsonify(
            reply    = composed,
            raw      = result['entry']['answer'],
            score    = result['score'],
            matched  = result['matched_kw'],
            source   = result['entry'].get('source',''),
            intent   = 'kb_match',
        )
    return jsonify(reply=fallback_text(), score=0.0, matched=[], source=None, intent='fallback')


# ══════════════════════════════════════════════════════════════════════════════
# API — SLM management (create / switch / rename / delete / export / import)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/slm', methods=['GET'])
def slm_list():
    reg = load_registry()
    active_id = get_active_id()
    result = []
    for s in reg['slms']:
        data = load_slm_data(s['id'])
        result.append({
            'id':        s['id'],
            'name':      s['name'],
            'created':   s.get('created', ''),
            'entries':   len(data.get('kb', [])),
            'documents': len(data.get('documents', [])),
            'active':    s['id'] == active_id,
        })
    return jsonify(result)


@app.route('/api/slm', methods=['POST'])
def slm_create():
    """Buat SLM baru yang KOSONG. Body JSON: { "name": "...", "activate": true }"""
    body = request.get_json(force=True) or {}
    name = (body.get('name') or '').strip() or f'SLM {now_str()}'
    activate = body.get('activate', True)

    reg = load_registry()
    slm_id = _new_slm_id()
    save_slm_data(slm_id, {'kb': [], 'documents': []})
    reg['slms'].append({'id': slm_id, 'name': name, 'created': now_str()})
    if activate:
        reg['active'] = slm_id
    save_registry(reg)
    return jsonify(id=slm_id, name=name, active=activate), 201


@app.route('/api/slm/<slm_id>', methods=['PUT'])
def slm_rename(slm_id):
    body = request.get_json(force=True) or {}
    name = (body.get('name') or '').strip()
    if not name:
        return jsonify(error='Nama wajib diisi'), 400
    reg = load_registry()
    for s in reg['slms']:
        if s['id'] == slm_id:
            s['name'] = name
            save_registry(reg)
            return jsonify(ok=True, id=slm_id, name=name)
    return jsonify(error='SLM tidak ditemukan'), 404


@app.route('/api/slm/<slm_id>', methods=['DELETE'])
def slm_delete(slm_id):
    reg = load_registry()
    if len(reg['slms']) <= 1:
        return jsonify(error='Tidak bisa menghapus satu-satunya SLM yang tersisa'), 400
    if not any(s['id'] == slm_id for s in reg['slms']):
        return jsonify(error='SLM tidak ditemukan'), 404

    reg['slms'] = [s for s in reg['slms'] if s['id'] != slm_id]
    if reg.get('active') == slm_id:
        reg['active'] = reg['slms'][0]['id']
    save_registry(reg)

    try:
        _slm_file(slm_id).unlink(missing_ok=True)
    except Exception:
        pass

    return jsonify(ok=True, active=reg['active'])


@app.route('/api/slm/<slm_id>/activate', methods=['POST'])
def slm_activate(slm_id):
    reg = load_registry()
    if not any(s['id'] == slm_id for s in reg['slms']):
        return jsonify(error='SLM tidak ditemukan'), 404
    reg['active'] = slm_id
    save_registry(reg)
    return jsonify(ok=True, active=slm_id)


@app.route('/api/slm/<slm_id>/export', methods=['GET'])
def slm_export(slm_id):
    """Download satu SLM (kb + documents) sebagai satu file JSON."""
    reg = load_registry()
    meta = next((s for s in reg['slms'] if s['id'] == slm_id), None)
    if not meta:
        return jsonify(error='SLM tidak ditemukan'), 404

    data = load_slm_data(slm_id)
    payload = {
        'format':      'csbot-slm',
        'version':     1,
        'id':          meta['id'],
        'name':        meta['name'],
        'exported_at': now_str(),
        'kb':          data.get('kb', []),
        'documents':   data.get('documents', []),
    }
    buf = io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))
    buf.seek(0)
    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', meta['name']).strip('_') or 'slm'
    fname = f'slm_{safe_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    return send_file(buf, mimetype='application/json',
                      as_attachment=True, download_name=fname)


@app.route('/api/slm/import', methods=['POST'])
def slm_import():
    """
    Import file JSON hasil export SLM (atau JSON kompatibel {"kb":[...],"documents":[...]})
    sebagai SLM BARU (tidak menimpa SLM yang sudah ada).
    POST multipart/form-data  file=<.json>  [name=...]
    atau JSON  { "slm": {...}, "name": "..." }
    """
    name_override = None
    payload = None

    if request.content_type and 'multipart' in request.content_type:
        f = request.files.get('file')
        if not f:
            return jsonify(error='Tidak ada file'), 400
        try:
            payload = json.loads(f.read().decode('utf-8'))
        except Exception:
            return jsonify(error='File JSON tidak valid'), 400
        name_override = request.form.get('name')
    else:
        body = request.get_json(force=True) or {}
        payload = body.get('slm')
        name_override = body.get('name')
        if payload is None:
            return jsonify(error='Field "slm" wajib diisi'), 400

    if not isinstance(payload, dict) or 'kb' not in payload:
        return jsonify(error='Format SLM JSON tidak valid (field "kb" tidak ditemukan)'), 422

    kb   = payload.get('kb') or []
    docs = payload.get('documents') or []
    if not isinstance(kb, list) or not isinstance(docs, list):
        return jsonify(error='Field "kb" dan "documents" harus berupa array'), 422

    # Regenerate id supaya tidak bentrok dengan SLM lain
    for e in kb:
        e['id'] = str(uuid.uuid4())

    reg = load_registry()
    slm_id = _new_slm_id()
    name = (name_override or payload.get('name') or 'Imported SLM').strip()
    save_slm_data(slm_id, {'kb': kb, 'documents': docs})
    reg['slms'].append({'id': slm_id, 'name': name, 'created': now_str()})
    reg['active'] = slm_id
    save_registry(reg)

    return jsonify(ok=True, id=slm_id, name=name, entries=len(kb), documents=len(docs)), 201


# ══════════════════════════════════════════════════════════════════════════════
# API — Knowledge Base CRUD (selalu terhadap SLM yang sedang aktif)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/kb', methods=['GET'])
def kb_list():
    return jsonify(load_kb())

@app.route('/api/kb', methods=['POST'])
def kb_add():
    body = request.get_json(force=True)
    keywords = [k.strip() for k in body.get('keywords', []) if k.strip()]
    answer   = (body.get('answer') or '').strip()
    if not keywords or not answer:
        return jsonify(error='keywords dan answer wajib diisi'), 400

    entry = {
        'id':       str(uuid.uuid4()),
        'keywords': keywords,
        'answer':   answer,
        'source':   body.get('source', 'manual'),
        'created':  now_str(),
    }
    kb = load_kb()
    kb.append(entry)
    save_kb(kb)
    return jsonify(entry), 201

@app.route('/api/kb/<entry_id>', methods=['PUT'])
def kb_update(entry_id):
    body = request.get_json(force=True)
    kb = load_kb()
    for e in kb:
        if e['id'] == entry_id:
            if 'keywords' in body:
                e['keywords'] = [k.strip() for k in body['keywords'] if k.strip()]
            if 'answer' in body:
                e['answer'] = body['answer'].strip()
            e['updated'] = now_str()
            save_kb(kb)
            return jsonify(e)
    return jsonify(error='Not found'), 404

@app.route('/api/kb/<entry_id>', methods=['DELETE'])
def kb_delete(entry_id):
    kb = load_kb()
    new_kb = [e for e in kb if e['id'] != entry_id]
    if len(new_kb) == len(kb):
        return jsonify(error='Not found'), 404
    save_kb(new_kb)
    return jsonify(ok=True)

@app.route('/api/kb/clear', methods=['POST'])
def kb_clear():
    source = request.get_json(force=True).get('source')
    kb = load_kb()
    if source:
        kb = [e for e in kb if e.get('source') != source]
    else:
        kb = []
    save_kb(kb)
    return jsonify(ok=True, remaining=len(kb))


# ══════════════════════════════════════════════════════════════════════════════
# API — Document Upload + NLP Processing (masuk ke SLM aktif)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/upload', methods=['POST'])
def upload_text():
    """
    POST multipart/form-data  file=<.txt>
    atau JSON  { "text": "...", "filename": "..." }

    Pipeline:
      1. Terima teks
      2. Split per paragraf
      3. Ekstraksi keyword TF-IDF per paragraf
      4. Simpan ke SLM yang sedang aktif  {id, keywords, answer, source}
      5. Catat dokumen di SLM yang sedang aktif
    """
    filename = 'upload'
    raw_text = ''

    if request.content_type and 'multipart' in request.content_type:
        f = request.files.get('file')
        if not f:
            return jsonify(error='Tidak ada file'), 400
        filename = f.filename
        if not filename.endswith('.txt'):
            return jsonify(error='Hanya file .txt yang didukung'), 400
        raw_text = f.read().decode('utf-8', errors='replace')
    else:
        body = request.get_json(force=True)
        raw_text = (body.get('text') or '').strip()
        filename = (body.get('filename') or 'paste').strip()

    if not raw_text:
        return jsonify(error='Teks kosong'), 400

    # Remove existing entries for this source
    kb = load_kb()
    kb = [e for e in kb if e.get('source') != filename]

    # NLP processing
    entries = parse_text_to_entries(raw_text, source=filename)
    if not entries:
        return jsonify(error='Tidak ada paragraf valid ditemukan (min 30 karakter per paragraf)'), 422

    new_entries = []
    for e in entries:
        entry = {
            'id':       str(uuid.uuid4()),
            'keywords': e['keywords'],
            'answer':   e['answer'],
            'source':   filename,
            'created':  now_str(),
        }
        new_entries.append(entry)

    kb.extend(new_entries)
    save_kb(kb)

    # Save document record
    docs = load_docs()
    docs = [d for d in docs if d['filename'] != filename]
    docs.append({
        'filename':   filename,
        'paragraphs': len(entries),
        'uploaded':   now_str(),
        'char_count': len(raw_text),
        'source_type': 'text',
    })
    save_docs(docs)

    return jsonify(
        ok          = True,
        filename    = filename,
        paragraphs  = len(entries),
        entries_added = len(new_entries),
        preview     = new_entries[:3],
    ), 201


@app.route('/api/upload/preview', methods=['POST'])
def preview_extraction():
    """
    Preview NLP extraction sebelum disimpan.
    POST JSON { "text": "...", "top_n": 8 }
    """
    body     = request.get_json(force=True)
    raw_text = (body.get('text') or '').strip()
    top_n    = min(int(body.get('top_n', 8)), 15)

    if not raw_text:
        return jsonify(error='Teks kosong'), 400

    raw_paras = re.split(r'\n\s*\n', raw_text.strip())
    paras = [p.strip() for p in raw_paras if len(p.strip()) >= 30]

    if not paras:
        return jsonify(error='Tidak ada paragraf valid'), 422

    kw_lists = extract_keywords_tfidf(paras, top_n=top_n)
    result = [
        {'paragraph': p[:200] + ('…' if len(p) > 200 else ''), 'keywords': kws}
        for p, kws in zip(paras, kw_lists)
    ]
    return jsonify(paragraphs=len(paras), extraction=result)


# ══════════════════════════════════════════════════════════════════════════════
# API — Web Scraping (masuk ke SLM aktif, hasil disimpan sebagai SATU dokumen)
# ══════════════════════════════════════════════════════════════════════════════
def _scrape_error_response(e):
    if isinstance(e, ValueError):
        return jsonify(error=str(e)), 422
    if isinstance(e, _requests.exceptions.RequestException):
        return jsonify(error=f'Gagal mengambil URL: {e}'), 502
    return jsonify(error=f'Gagal scraping: {e}'), 500


@app.route('/api/scrape/preview', methods=['POST'])
def scrape_preview():
    """
    Preview hasil scraping tanpa menyimpan.
    POST JSON { "url": "...", "tag": "p", "css_class": "content", "selector": "" }
    """
    body      = request.get_json(force=True) or {}
    url       = (body.get('url') or '').strip()
    tag       = body.get('tag')
    css_class = body.get('css_class')
    selector  = body.get('selector')

    if not url:
        return jsonify(error='URL wajib diisi'), 400

    try:
        paragraphs, used_selector = scrape_url(url, tag, css_class, selector)
    except (ValueError, _requests.exceptions.RequestException) as e:
        return _scrape_error_response(e)
    except Exception as e:
        return _scrape_error_response(e)

    paras = [p for p in paragraphs if len(p) >= 30]
    if not paras:
        return jsonify(error='Teks hasil scraping terlalu pendek (min 30 karakter per elemen)'), 422

    kw_lists = extract_keywords_tfidf(paras, top_n=8)
    result = [
        {'paragraph': p[:200] + ('…' if len(p) > 200 else ''), 'keywords': kws}
        for p, kws in zip(paras, kw_lists)
    ]
    return jsonify(
        url            = url,
        selector       = used_selector,
        elements_found = len(paragraphs),
        paragraphs     = len(paras),
        extraction     = result,
    )


@app.route('/api/scrape', methods=['POST'])
def scrape_save():
    """
    Ambil halaman, ekstrak teks dari elemen yang cocok dengan tag/class/selector,
    lalu proses lewat pipeline NLP yang sama dengan upload dokumen. Semua elemen
    yang cocok pada satu URL digabung dan disimpan sebagai SATU dokumen sumber.

    POST JSON { "url": "...", "tag": "p", "css_class": "content",
                "selector": "", "filename": "nama-opsional" }
    """
    body      = request.get_json(force=True) or {}
    url       = (body.get('url') or '').strip()
    tag       = body.get('tag')
    css_class = body.get('css_class')
    selector  = body.get('selector')
    filename  = (body.get('filename') or '').strip()

    if not url:
        return jsonify(error='URL wajib diisi'), 400

    try:
        paragraphs, used_selector = scrape_url(url, tag, css_class, selector)
    except (ValueError, _requests.exceptions.RequestException) as e:
        return _scrape_error_response(e)
    except Exception as e:
        return _scrape_error_response(e)

    if not filename:
        parsed = urlparse(url)
        slug = re.sub(r'[^A-Za-z0-9]+', '_', (parsed.netloc + parsed.path)).strip('_')
        filename = f'scrape_{slug[:50]}' if slug else f'scrape_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if not filename.endswith('.txt'):
        filename += '.txt'

    # Gabung semua elemen jadi satu blok teks (satu elemen = satu paragraf)
    raw_text = '\n\n'.join(paragraphs)

    # Replace entries lama dari source/filename yang sama (perilaku sama seperti upload)
    kb = load_kb()
    kb = [e for e in kb if e.get('source') != filename]

    entries = parse_text_to_entries(raw_text, source=filename)
    if not entries:
        return jsonify(error='Tidak ada paragraf valid ditemukan dari hasil scraping (min 30 karakter)'), 422

    new_entries = []
    for e in entries:
        entry = {
            'id':       str(uuid.uuid4()),
            'keywords': e['keywords'],
            'answer':   e['answer'],
            'source':   filename,
            'created':  now_str(),
        }
        new_entries.append(entry)

    kb.extend(new_entries)
    save_kb(kb)

    # Simpan sebagai SATU dokumen sumber (dengan metadata asal scraping)
    docs = load_docs()
    docs = [d for d in docs if d['filename'] != filename]
    docs.append({
        'filename':    filename,
        'paragraphs':  len(entries),
        'uploaded':    now_str(),
        'char_count':  len(raw_text),
        'source_type': 'scrape',
        'source_url':  url,
        'selector':    used_selector,
    })
    save_docs(docs)

    return jsonify(
        ok             = True,
        filename       = filename,
        url            = url,
        selector       = used_selector,
        elements_found = len(paragraphs),
        paragraphs     = len(entries),
        entries_added  = len(new_entries),
        preview        = new_entries[:3],
    ), 201


# ══════════════════════════════════════════════════════════════════════════════
# API — Documents list
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/documents', methods=['GET'])
def doc_list():
    return jsonify(load_docs())

@app.route('/api/documents/<filename>', methods=['DELETE'])
def doc_delete(filename):
    kb = load_kb()
    kb = [e for e in kb if e.get('source') != filename]
    save_kb(kb)
    docs = load_docs()
    docs = [d for d in docs if d['filename'] != filename]
    save_docs(docs)
    return jsonify(ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# API — Decision Tree PNG
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/tree/png')
def tree_png():
    """
    Render full decision tree as PNG.
    Query params:
      dpi=<int>      default 130
      download=1     add Content-Disposition: attachment
    """
    try:
        dpi      = min(int(request.args.get('dpi', 130)), 300)
        download = request.args.get('download', '0') == '1'
        kb       = load_kb()
        png_bytes = build_tree_png(kb, dpi=dpi)
        buf = io.BytesIO(png_bytes)
        buf.seek(0)
        if download:
            return send_file(
                buf,
                mimetype='image/png',
                as_attachment=True,
                download_name='decision_tree.png',
            )
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify(error=str(e)), 500


# ══════════════════════════════════════════════════════════════════════════════
# API — Stats
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/stats', methods=['GET'])
def stats():
    kb   = load_kb()
    docs = load_docs()
    sources = {}
    for e in kb:
        s = e.get('source','manual')
        sources[s] = sources.get(s, 0) + 1
    return jsonify(
        total_entries = len(kb),
        total_docs    = len(docs),
        sources       = sources,
    )


# ══════════════════════════════════════════════════════════════════════════════
# API — Graph Database (JSON export / import + visual analytics)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/graph/data', methods=['GET'])
def graph_data():
    """
    Return the KB projected as a graph for visualization.
    Query param: mode = hierarchy (default) | keyword_network
    """
    mode = request.args.get('mode', 'hierarchy')
    kb = load_kb()
    graph = build_keyword_network(kb) if mode == 'keyword_network' else build_graph(kb)
    return jsonify(graph)


@app.route('/api/graph/export', methods=['GET'])
def graph_export():
    """
    Download the KB as a JSON graph file.
    Query param: mode = hierarchy (default, importable) | keyword_network (analysis only)
    """
    mode = request.args.get('mode', 'hierarchy')
    kb = load_kb()
    graph = build_keyword_network(kb) if mode == 'keyword_network' else build_graph(kb)
    payload = json.dumps(graph, ensure_ascii=False, indent=2).encode('utf-8')
    buf = io.BytesIO(payload)
    buf.seek(0)
    fname = f'graph_{mode}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    return send_file(buf, mimetype='application/json',
                      as_attachment=True, download_name=fname)


@app.route('/api/graph/import', methods=['POST'])
def graph_import():
    """
    Import a hierarchy graph JSON back into the knowledge base (SLM aktif).
    POST multipart/form-data  file=<.json>  [mode=replace|merge]
    atau JSON  { "graph": {...}, "mode": "replace|merge" }
    """
    mode = request.args.get('mode') or 'replace'
    graph = None

    if request.content_type and 'multipart' in request.content_type:
        f = request.files.get('file')
        if not f:
            return jsonify(error='Tidak ada file'), 400
        try:
            graph = json.loads(f.read().decode('utf-8'))
        except Exception:
            return jsonify(error='File JSON tidak valid'), 400
        mode = request.form.get('mode', mode)
    else:
        body = request.get_json(force=True) or {}
        graph = body.get('graph')
        mode  = body.get('mode', mode)
        if graph is None:
            return jsonify(error='Field "graph" wajib diisi'), 400

    try:
        new_entries = kb_from_graph(graph)
    except ValueError as e:
        return jsonify(error=str(e)), 422

    if mode == 'merge':
        kb = load_kb()
        existing_ids = {e['id'] for e in kb}
        for e in new_entries:
            if e['id'] in existing_ids:
                e['id'] = str(uuid.uuid4())
            existing_ids.add(e['id'])
            kb.append(e)
        save_kb(kb)
        _rebuild_docs_from_kb(kb)
        return jsonify(ok=True, mode='merge', imported=len(new_entries), total=len(kb)), 201

    save_kb(new_entries)
    _rebuild_docs_from_kb(new_entries)
    return jsonify(ok=True, mode='replace', imported=len(new_entries), total=len(new_entries)), 201


def _rebuild_docs_from_kb(kb: list):
    """Regenerate documents.json counts to stay consistent after a graph import."""
    counts = {}
    for e in kb:
        src = e.get('source', 'manual')
        counts[src] = counts.get(src, 0) + 1
    docs = [
        {'filename': s, 'paragraphs': c, 'uploaded': now_str(), 'char_count': 0}
        for s, c in counts.items()
    ]
    save_docs(docs)


@app.route('/api/graph/stats', methods=['GET'])
def graph_stats_route():
    kb = load_kb()
    return jsonify(graph_stats(kb))


if __name__ == '__main__':
    # Pastikan registry SLM sudah ada (memicu migrasi data lama kalau perlu)
    load_registry()

    # Seed default KB kalau SLM aktif masih benar-benar kosong
    if not load_kb():
        defaults = [
            {'keywords':['jam','buka','operasional','tutup','kerja'],
             'answer':'Kami beroperasi Senin–Jumat pukul 08.00–17.00 WIB dan Sabtu 08.00–12.00 WIB. Hari Minggu dan libur nasional kami tutup.'},
            {'keywords':['hubungi','kontak','telepon','email','whatsapp'],
             'answer':'📞 Telepon: (0370) 123-4567\n📧 Email: cs@example.ac.id\n💬 WhatsApp: 0812-3456-7890'},
            {'keywords':['harga','biaya','tarif','bayar','cost'],
             'answer':'Informasi harga tersedia di website kami. Harga berbeda tergantung layanan yang dipilih. Hubungi kami untuk penawaran khusus.'},
            {'keywords':['daftar','registrasi','mendaftar','pendaftaran'],
             'answer':'Pendaftaran dapat dilakukan secara online melalui website resmi atau langsung ke kantor kami dengan membawa dokumen yang diperlukan.'},
            {'keywords':['status','cek','lacak','pesanan','order'],
             'answer':'Untuk mengecek status, login ke akun Anda atau hubungi CS dengan menyebutkan nomor transaksi/referensi Anda.'},
        ]
        kb = []
        for d in defaults:
            kb.append({'id': str(uuid.uuid4()), 'source': 'default',
                        'created': now_str(), **d})
        save_kb(kb)
        print(f'[SEED] {len(kb)} default entries created into active SLM.')

    app.run(debug=True, port=5000)
