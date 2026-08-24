"""
graph_engine.py — Graph-database layer for the CS Bot knowledge base.

The knowledge base (list of {id, keywords, answer, source, created}) is
projected into a JSON graph of nodes + edges. Two graph "kinds" are
supported:

  hierarchy         root → source → entry → keyword
                     Lossless — can be exported AND re-imported to
                     rebuild the knowledge base exactly.

  keyword_network   keyword ↔ keyword, edge weight = co-occurrence count
                     Analytical view only (not importable) — used to spot
                     clusters of related topics and bridge keywords.

Also provides graph_stats() for the visual analytics sidebar.
"""

import datetime
import uuid
from collections import defaultdict, Counter
from itertools import combinations
from typing import List, Dict, Any


def _now() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _kw_id(kw: str) -> str:
    return f'kw::{kw.strip().lower()}'


def _src_id(src: str) -> str:
    return f'src::{src}'


# ══════════════════════════════════════════════════════════════════════════
# BUILD — hierarchy graph (lossless, importable)
# ══════════════════════════════════════════════════════════════════════════
def build_graph(kb: List[Dict]) -> Dict[str, Any]:
    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []

    nodes['root'] = {
        'id': 'root', 'type': 'root', 'label': 'CS Bot',
        'meta': {'entries': len(kb)},
    }

    sources: Dict[str, List[Dict]] = {}
    for e in kb:
        sources.setdefault(e.get('source', 'manual'), []).append(e)

    for src, entries in sources.items():
        sid = _src_id(src)
        nodes[sid] = {
            'id': sid, 'type': 'source', 'label': src,
            'meta': {'count': len(entries)},
        }
        edges.append({'from': 'root', 'to': sid, 'type': 'root_source'})

        for e in entries:
            eid = e['id']
            answer = e.get('answer', '')
            nodes[eid] = {
                'id': eid, 'type': 'entry',
                'label': (answer[:40] + '…') if len(answer) > 40 else answer,
                'meta': {
                    'keywords': e.get('keywords', []),
                    'answer':   answer,
                    'source':   src,
                    'created':  e.get('created', ''),
                    'updated':  e.get('updated', ''),
                },
            }
            edges.append({'from': sid, 'to': eid, 'type': 'source_entry'})

            for kw in e.get('keywords', []):
                kid = _kw_id(kw)
                if kid not in nodes:
                    nodes[kid] = {'id': kid, 'type': 'keyword', 'label': kw, 'meta': {'degree': 0}}
                nodes[kid]['meta']['degree'] += 1
                edges.append({'from': eid, 'to': kid, 'type': 'entry_keyword'})

    return {
        'version':     1,
        'kind':        'hierarchy',
        'exported_at': _now(),
        'nodes':       list(nodes.values()),
        'edges':       edges,
    }


# ══════════════════════════════════════════════════════════════════════════
# BUILD — keyword co-occurrence network (analytical only)
# ══════════════════════════════════════════════════════════════════════════
def build_keyword_network(kb: List[Dict]) -> Dict[str, Any]:
    freq: Counter = Counter()
    cooc: Counter = Counter()
    kw_sources = defaultdict(set)

    for e in kb:
        kws = sorted({k.lower() for k in e.get('keywords', [])})
        src = e.get('source', 'manual')
        for k in kws:
            freq[k] += 1
            kw_sources[k].add(src)
        for a, b in combinations(kws, 2):
            cooc[(a, b)] += 1

    nodes = [
        {
            'id': _kw_id(k), 'type': 'keyword', 'label': k,
            'meta': {'frequency': c, 'sources': sorted(kw_sources[k])},
        }
        for k, c in freq.items()
    ]
    edges = [
        {'from': _kw_id(a), 'to': _kw_id(b), 'type': 'co_occurs', 'weight': w}
        for (a, b), w in cooc.items()
    ]

    return {
        'version':     1,
        'kind':        'keyword_network',
        'exported_at': _now(),
        'nodes':       nodes,
        'edges':       edges,
    }


# ══════════════════════════════════════════════════════════════════════════
# STATS — analytics for the graph sidebar
# ══════════════════════════════════════════════════════════════════════════
def graph_stats(kb: List[Dict]) -> Dict[str, Any]:
    if not kb:
        return {
            'total_entries': 0, 'total_sources': 0, 'total_keywords': 0,
            'total_edges': 0, 'avg_keywords_per_entry': 0, 'density': 0,
            'top_keywords': [], 'entries_per_source': {},
            'isolated_keywords': [], 'isolated_keyword_count': 0,
            'top_cooccurring_pairs': [], 'entries_without_keywords': 0,
        }

    freq: Counter = Counter()
    cooc: Counter = Counter()
    entries_per_source: Counter = Counter()
    kw_total_len = 0
    entries_without_kw = 0

    for e in kb:
        src = e.get('source', 'manual')
        entries_per_source[src] += 1
        kws = sorted({k.lower() for k in e.get('keywords', [])})
        kw_total_len += len(kws)
        if not kws:
            entries_without_kw += 1
        for k in kws:
            freq[k] += 1
        for a, b in combinations(kws, 2):
            cooc[(a, b)] += 1

    n_entries  = len(kb)
    n_keywords = len(freq)
    n_sources  = len(entries_per_source)

    # root→source + source→entry + entry→keyword edges
    total_edges = n_sources + n_entries + kw_total_len

    max_possible = n_entries * n_keywords if n_keywords else 1
    density = round(kw_total_len / max_possible, 4) if max_possible else 0

    isolated = [k for k, c in freq.items() if c == 1]

    return {
        'total_entries':   n_entries,
        'total_sources':   n_sources,
        'total_keywords':  n_keywords,
        'total_edges':     total_edges,
        'avg_keywords_per_entry':  round(kw_total_len / n_entries, 2),
        'density':                 density,
        'top_keywords':            [{'keyword': k, 'count': c} for k, c in freq.most_common(15)],
        'entries_per_source':      dict(entries_per_source),
        'isolated_keywords':       isolated[:20],
        'isolated_keyword_count':  len(isolated),
        'top_cooccurring_pairs':   [{'a': a, 'b': b, 'count': c} for (a, b), c in cooc.most_common(10)],
        'entries_without_keywords': entries_without_kw,
    }


# ══════════════════════════════════════════════════════════════════════════
# IMPORT — rebuild KB entries from a hierarchy graph
# ══════════════════════════════════════════════════════════════════════════
def kb_from_graph(graph: Dict[str, Any]) -> List[Dict]:
    if not isinstance(graph, dict):
        raise ValueError('Graph harus berupa objek JSON')

    nodes = graph.get('nodes')
    if not nodes:
        raise ValueError('Graph tidak memiliki node')

    entry_nodes = [n for n in nodes if n.get('type') == 'entry']
    if not entry_nodes:
        raise ValueError('Graph tidak mengandung node bertipe "entry" — gunakan graph kind="hierarchy"')

    kb = []
    seen_ids = set()
    for n in entry_nodes:
        meta = n.get('meta') or {}
        keywords = meta.get('keywords')
        answer   = meta.get('answer')
        if not keywords or not answer:
            continue

        eid = n.get('id') or str(uuid.uuid4())
        if eid in seen_ids:
            eid = str(uuid.uuid4())
        seen_ids.add(eid)

        entry = {
            'id':       eid,
            'keywords': [str(k).strip() for k in keywords if str(k).strip()],
            'answer':   str(answer).strip(),
            'source':   meta.get('source', 'manual'),
            'created':  meta.get('created') or _now(),
        }
        if meta.get('updated'):
            entry['updated'] = meta['updated']
        kb.append(entry)

    if not kb:
        raise ValueError('Tidak ada entry valid untuk direkonstruksi dari graph')
    return kb
