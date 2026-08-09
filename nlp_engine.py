"""
nlp_engine.py — Indonesian NLP: keyword extraction + decision tree matching
Proses: tokenisasi → stopword removal → stemming → TF-IDF → top keywords
"""

import re
import math
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional

# ── Indonesian stopwords (extended) ──────────────────────────────────────────
STOPWORDS_ID = {
    'yang','dan','di','ke','dari','ini','itu','dengan','untuk','adalah','pada',
    'dalam','tidak','ada','atau','juga','akan','saya','kami','kita','mereka',
    'ia','dia','anda','bisa','dapat','telah','sudah','harus','lebih','jika',
    'bila','karena','oleh','saat','ketika','namun','tetapi','tapi','namun',
    'bahwa','seperti','agar','supaya','maka','lalu','kemudian','setelah',
    'sebelum','antara','bagi','per','hal','cara','semua','setiap','banyak',
    'beberapa','salah','satu','dua','tiga','empat','lima','enam','tujuh',
    'delapan','sembilan','sepuluh','serta','selain','maupun','sehingga',
    'sedangkan','adapun','baik','atas','bawah','kiri','kanan','mana','dimana',
    'bagaimana','kapan','mengapa','siapa','apakah','apa','ya','tidak','no',
    'ok','oke','oh','ah','eh','hmm','nya','lah','kah','pun','pula','saja',
    'hanya','cuma','sangat','sekali','sangat','amat','ter','me','di','ber',
    'paling','lebih','kurang','sama','jadi','menjadi','tentang','mengenai',
    'berkaitan','terkait','sesuai','setelah','kami','kita','mereka','nya',
    'lainnya','lain','seluruh','semua','masing','tiap','setiap','berbagai',
}

# ── Light Indonesian stemmer (Nazief-Adriani simplified) ─────────────────────
PREFIXES = ['menge','memper','mempel','mempe','membe','memba','membo',
            'menye','menge','penge','peny','meny','pen','mem','men','meng',
            'me','ber','per','ke','ter','se','di','pe']
SUFFIXES = ['kan','nya','lah','kah','an','i']
INFIXES  = ['el','em','er']

def stem_id(word: str) -> str:
    """Simplified Indonesian stemmer."""
    if len(word) <= 3:
        return word
    w = word.lower()
    # strip suffix
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3:
            w = w[:-len(s)]
            break
    # strip prefix
    for p in PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    return w if len(w) >= 3 else word.lower()


def tokenize(text: str) -> List[str]:
    """Tokenize and clean text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS_ID]


def stem_tokens(tokens: List[str]) -> List[str]:
    return [stem_id(t) for t in tokens]


def extract_keywords_tfidf(paragraphs: List[str], top_n: int = 8) -> List[List[str]]:
    """
    Per-paragraph TF-IDF keyword extraction.
    Returns list of keyword lists, one per paragraph.
    """
    # Build token lists per paragraph
    docs = []
    for para in paragraphs:
        tokens = tokenize(para)
        tokens = remove_stopwords(tokens)
        tokens = stem_tokens(tokens)
        # Also include original tokens (unstemmed) for display
        orig = remove_stopwords(tokenize(para))
        docs.append({'stems': tokens, 'orig': orig})

    N = len(docs)
    if N == 0:
        return []

    # Document frequency (on stems)
    df: Counter = Counter()
    for doc in docs:
        unique_stems = set(doc['stems'])
        df.update(unique_stems)

    results = []
    for doc in docs:
        tf: Counter = Counter(doc['stems'])
        total = max(len(doc['stems']), 1)

        scores: Dict[str, float] = {}
        for stem, count in tf.items():
            tf_val  = count / total
            idf_val = math.log((N + 1) / (df[stem] + 1)) + 1
            scores[stem] = tf_val * idf_val

        # Map stems back to best original token
        stem_to_orig: Dict[str, str] = {}
        for orig_tok in doc['orig']:
            s = stem_id(orig_tok)
            if s not in stem_to_orig:
                stem_to_orig[s] = orig_tok

        top_stems = sorted(scores, key=scores.get, reverse=True)[:top_n]
        keywords = []
        seen = set()
        for s in top_stems:
            kw = stem_to_orig.get(s, s)
            if kw not in seen and kw not in STOPWORDS_ID:
                keywords.append(kw)
                seen.add(kw)

        results.append(keywords)

    return results


def parse_text_to_entries(text: str, source: str = '') -> List[Dict]:
    """
    Split text into paragraphs, extract keywords per paragraph,
    return list of {keywords, answer, source} dicts.
    """
    # Split by blank lines (paragraph boundary)
    raw_paras = re.split(r'\n\s*\n', text.strip())
    paras = [p.strip() for p in raw_paras if len(p.strip()) >= 30]

    if not paras:
        return []

    kw_lists = extract_keywords_tfidf(paras, top_n=8)

    entries = []
    for para, kws in zip(paras, kw_lists):
        if kws:
            entries.append({
                'keywords': kws,
                'answer':   para.replace('\n', ' ').strip(),
                'source':   source,
            })
    return entries


# ── Decision Tree Matching ────────────────────────────────────────────────────

def _token_match(a: str, b: str) -> bool:
    """Fuzzy token match: exact, prefix, or suffix overlap ≥3 chars."""
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3:
        if a.startswith(b) or b.startswith(a):
            return True
        if a.endswith(b[-3:]) or b.endswith(a[-3:]):
            return True
    return False


def score_entry(entry: Dict, query_stems: List[str], query_orig: List[str]) -> float:
    """Score a KB entry against query tokens (stems + originals)."""
    kw_stems = [stem_id(k) for k in entry['keywords']]
    kw_orig  = [k.lower() for k in entry['keywords']]

    # Also score against the full answer text (lightweight)
    ans_stems = stem_tokens(remove_stopwords(tokenize(entry['answer'])))

    hits_kw_stem = sum(1 for qs in query_stems if any(_token_match(qs, ks) for ks in kw_stems))
    hits_kw_orig = sum(1 for qo in query_orig  if any(_token_match(qo, ko) for ko in kw_orig))
    hits_ans     = sum(1 for qs in query_stems if any(_token_match(qs, a)  for a  in ans_stems))

    kw_hits  = max(hits_kw_stem, hits_kw_orig)
    ans_hits = hits_ans

    if kw_hits == 0 and ans_hits == 0:
        return 0.0

    kw_cov   = kw_hits  / max(len(kw_stems), 1)
    kw_prec  = kw_hits  / max(len(query_stems), 1)
    ans_prec = ans_hits / max(len(query_stems), 1)

    # Primary: keyword match; secondary: answer text match (lower weight)
    return kw_cov * 0.45 + kw_prec * 0.35 + ans_prec * 0.20


def query_kb(knowledge_base: List[Dict], user_text: str,
             threshold: float = 0.08) -> Optional[Dict]:
    """
    Run decision-tree matching against knowledge base.
    Returns best match dict or None.
    """
    if not knowledge_base:
        return None

    tokens     = tokenize(user_text)
    orig_clean = remove_stopwords(tokens)
    stems      = stem_tokens(orig_clean)

    best_entry, best_score = None, 0.0
    matched_kws = []

    for entry in knowledge_base:
        score = score_entry(entry, stems, orig_clean)
        if score > best_score:
            best_score = score
            best_entry = entry
            # collect which keywords matched
            kw_stems = [stem_id(k) for k in entry['keywords']]
            matched_kws = [
                kw for kw, ks in zip(entry['keywords'], kw_stems)
                if any(qs == ks or qs.startswith(ks) or ks.startswith(qs)
                       for qs in stems)
            ]

    if best_score < threshold or best_entry is None:
        return None

    return {
        'entry':      best_entry,
        'score':      round(best_score, 4),
        'matched_kw': matched_kws,
    }
