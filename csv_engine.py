"""
csv_engine.py — Modul upload & analisis CSV menggunakan pandas.

Alur:
  1. Baca CSV — baris pertama otomatis dipakai sebagai nama kolom
     (perilaku default pandas: header=0).
  2. Deteksi tipe tiap kolom: numerik, tanggal/waktu, atau kategorikal/teks.
  3. Hitung statistik: describe() (count, mean, std, min, kuartil, max),
     correlation matrix antar kolom numerik, serta statistik kolom
     kategorikal (nilai unik, modus, dsb).
  4. Susun semua hasil di atas menjadi kalimat Bahasa Indonesia yang natural,
     lalu bungkus jadi entri Knowledge Base {keywords, answer} — sengaja
     memakai bentuk yang sama seperti nlp_engine.parse_text_to_entries()
     supaya bisa langsung disimpan ke SLM aktif dan dijawab lewat pipeline
     /api/chat + response_composer.py yang sudah ada (tanpa mengubah kedua
     modul itu, tetap 100% lokal/offline, konsisten dengan filosofi
     decision-tree csbot).
  5. Untuk kolom numerik, kalimat nilai tertinggi/terendah otomatis
     menyertakan tanggal kejadian kalau ada kolom tanggal/waktu di baris
     yang sama — mis. "Suhu tertinggi adalah 38.9 pada tanggal 19 Agustus
     2026."
"""

import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Nama bulan Indonesia ──────────────────────────────────────────────────
BULAN_ID = [
    'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
    'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember',
]

# Petunjuk nama kolom yang kemungkinan berisi tanggal/waktu
DATE_NAME_HINTS = [
    'tanggal', 'tgl', 'date', 'waktu', 'time', 'jam',
    'periode', 'bulan', 'tahun', 'hari',
]

AGG_LABELS = {
    'max': 'tertinggi',
    'min': 'terendah',
    'mean': 'rata-rata',
    'sum': 'total',
}


# ══════════════════════════════════════════════════════════════════════════
# 1. Baca CSV
# ══════════════════════════════════════════════════════════════════════════
def load_csv(file_like_or_path, **kwargs) -> pd.DataFrame:
    """
    Baca file CSV. Baris pertama otomatis jadi nama kolom (header=0, default
    pandas). Mencoba beberapa delimiter umum (koma, titik koma) kalau
    delimiter tidak ditentukan secara eksplisit.
    """
    if 'sep' in kwargs or 'delimiter' in kwargs:
        df = pd.read_csv(file_like_or_path, **kwargs)
    else:
        try:
            df = pd.read_csv(file_like_or_path, sep=None, engine='python', **kwargs)
        except Exception:
            if hasattr(file_like_or_path, 'seek'):
                file_like_or_path.seek(0)
            df = pd.read_csv(file_like_or_path, **kwargs)

    df.columns = [str(c).strip() for c in df.columns]
    # Buang kolom "Unnamed: N" kosong yang sering muncul dari CSV berindeks
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed(:\s*\d+)?$', na=False)]
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. Deteksi tipe kolom
# ══════════════════════════════════════════════════════════════════════════
def _to_datetime_series(series: pd.Series) -> pd.Series:
    """
    Konversi Series ke datetime. Coba format standar (ISO dsb.) dulu, baru
    fallback ke dayfirst=True untuk format ambigu seperti '19/08/2026'.
    Warning parsing bawaan pandas disupres karena sudah ditangani manual.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().sum() < series.dropna().shape[0]:
            parsed_dayfirst = pd.to_datetime(series, errors='coerce', dayfirst=True)
            if parsed_dayfirst.notna().sum() > parsed.notna().sum():
                parsed = parsed_dayfirst
    return parsed


def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    """
    Coba parse kolom sebagai datetime. Kembalikan Series hasil parse kalau
    berhasil untuk mayoritas (>=80%) nilai non-kosong, kalau tidak None.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None
    try:
        parsed = _to_datetime_series(series)
    except Exception:
        return None
    success_rate = parsed.notna().sum() / max(len(non_null), 1)
    return parsed if success_rate >= 0.8 else None


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Kembalikan dict {nama_kolom: 'numeric' | 'datetime' | 'categorical'}.
    """
    types: Dict[str, str] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = 'datetime'
        elif pd.api.types.is_numeric_dtype(series):
            types[col] = 'numeric'
        elif _try_parse_datetime(series) is not None:
            types[col] = 'datetime'
        else:
            types[col] = 'categorical'
    return types


def coerce_datetime_columns(df: pd.DataFrame, types: Dict[str, str]) -> pd.DataFrame:
    """Kembalikan copy df dengan kolom bertipe 'datetime' dikonversi ke datetime64."""
    df2 = df.copy()
    for col, t in types.items():
        if t == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = _to_datetime_series(df2[col])
    return df2


def _primary_datetime_col(types: Dict[str, str]) -> Optional[str]:
    """Pilih kolom tanggal/waktu utama: prioritaskan nama yang mengandung
    petunjuk tanggal, kalau tidak ada pakai kolom datetime pertama."""
    dt_cols = [c for c, t in types.items() if t == 'datetime']
    if not dt_cols:
        return None
    for c in dt_cols:
        if any(h in c.lower() for h in DATE_NAME_HINTS):
            return c
    return dt_cols[0]


# ══════════════════════════════════════════════════════════════════════════
# 3. Statistik (pandas)
# ══════════════════════════════════════════════════════════════════════════
def compute_describe(df: pd.DataFrame, types: Dict[str, str]) -> pd.DataFrame:
    """describe() standar pandas untuk semua kolom numerik."""
    numeric_cols = [c for c, t in types.items() if t == 'numeric']
    if not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].describe()


def compute_correlation(df: pd.DataFrame, types: Dict[str, str]) -> pd.DataFrame:
    """Correlation matrix (Pearson) antar kolom numerik."""
    numeric_cols = [c for c, t in types.items() if t == 'numeric']
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(numeric_only=True)


def compute_categorical_stats(df: pd.DataFrame, types: Dict[str, str]) -> Dict[str, Dict]:
    """Statistik ringkas kolom kategorikal: jumlah unik, modus, missing."""
    result = {}
    for col, t in types.items():
        if t != 'categorical':
            continue
        vc = df[col].value_counts(dropna=True)
        result[col] = {
            'unique':  int(df[col].nunique(dropna=True)),
            'top':     (str(vc.index[0]) if not vc.empty else None),
            'top_count': int(vc.iloc[0]) if not vc.empty else 0,
            'missing': int(df[col].isna().sum()),
        }
    return result


def _fmt_num(x) -> str:
    """Format angka biar rapi di kalimat: bilangan bulat tanpa .0, desimal max 2 digit."""
    if x is None:
        return '-'
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf):
        return '-'
    if xf == int(xf):
        return str(int(xf))
    return f"{round(xf, 2)}"


def format_tanggal_id(ts) -> str:
    """
    Format Timestamp jadi 'DD Bulan YYYY' Bahasa Indonesia. Sertakan jam
    kalau time-of-day-nya bukan 00:00:00 (berarti kolomnya memang berisi jam).
    """
    if ts is None or pd.isna(ts):
        return ''
    ts = pd.Timestamp(ts)
    tanggal = f"{ts.day} {BULAN_ID[ts.month - 1]} {ts.year}"
    if ts.hour or ts.minute or ts.second:
        tanggal += f" pukul {ts.strftime('%H:%M')}"
    return tanggal


# ══════════════════════════════════════════════════════════════════════════
# 4. Kalimat Bahasa Indonesia
# ══════════════════════════════════════════════════════════════════════════
def dataset_overview_sentence(df: pd.DataFrame, types: Dict[str, str], source_name: str = '') -> str:
    """Ringkasan umum dataset: jumlah baris/kolom + kolom per tipe."""
    n_rows, n_cols = df.shape
    numeric_cols     = [c for c, t in types.items() if t == 'numeric']
    datetime_cols    = [c for c, t in types.items() if t == 'datetime']
    categorical_cols = [c for c, t in types.items() if t == 'categorical']

    label = f'"{source_name}" ' if source_name else ''
    parts = [f"Dataset {label}berisi {n_rows} baris data dan {n_cols} kolom."]
    if numeric_cols:
        parts.append(f"Kolom numerik: {', '.join(numeric_cols)}.")
    if datetime_cols:
        parts.append(f"Kolom tanggal/waktu: {', '.join(datetime_cols)}.")
    if categorical_cols:
        parts.append(f"Kolom kategorikal: {', '.join(categorical_cols)}.")
    return ' '.join(parts)


def describe_sentence(df: pd.DataFrame, types: Dict[str, str], column: str) -> Optional[str]:
    """Ringkasan describe() satu kolom numerik dalam satu kalimat."""
    if types.get(column) != 'numeric':
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    d = series.describe()
    return (
        f"Statistik deskriptif untuk {column}: jumlah data {int(d['count'])}, "
        f"rata-rata {_fmt_num(d['mean'])}, standar deviasi {_fmt_num(d['std'])}, "
        f"nilai minimum {_fmt_num(d['min'])}, kuartil 1 {_fmt_num(d['25%'])}, "
        f"median {_fmt_num(d['50%'])}, kuartil 3 {_fmt_num(d['75%'])}, "
        f"dan nilai maksimum {_fmt_num(d['max'])}."
    )


def extreme_value_sentence(
    df: pd.DataFrame,
    types: Dict[str, str],
    column: str,
    agg: str = 'max',
    datetime_col: Optional[str] = None,
) -> Optional[str]:
    """
    Kalimat Bahasa Indonesia untuk nilai tertinggi/terendah/rata-rata/total
    satu kolom numerik. Untuk agg 'max'/'min', kalau ada kolom tanggal/waktu
    (datetime_col, atau dideteksi otomatis), tanggal kejadiannya disertakan.

    Contoh: "Suhu tertinggi adalah 38.9 pada tanggal 19 Agustus 2026."
    """
    if types.get(column) != 'numeric':
        return None
    series = df[column].dropna()
    if series.empty:
        return None

    col_label = column[0].upper() + column[1:] if column else column
    label = AGG_LABELS.get(agg, agg)

    if agg == 'mean':
        return f"Rata-rata {column} adalah {_fmt_num(series.mean())}."
    if agg == 'sum':
        return f"Total {column} adalah {_fmt_num(series.sum())}."
    if agg not in ('max', 'min'):
        return None

    idx = series.idxmax() if agg == 'max' else series.idxmin()
    value_str = _fmt_num(df.loc[idx, column])

    dt_col = datetime_col or _primary_datetime_col(types)
    if dt_col and dt_col in df.columns:
        ts = df.loc[idx, dt_col]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts, errors='coerce', dayfirst=True)
        tanggal_str = format_tanggal_id(ts)
        if tanggal_str:
            return f"{col_label} {label} adalah {value_str} pada tanggal {tanggal_str}."

    return f"{col_label} {label} adalah {value_str}."


def correlation_sentences(df: pd.DataFrame, types: Dict[str, str], top_n: int = 5) -> List[str]:
    """Kalimat naratif pasangan kolom numerik dengan korelasi terkuat (mutlak),
    diurutkan dari yang paling kuat."""
    corr = compute_correlation(df, types)
    if corr.empty:
        return []
    cols = corr.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            r = corr.loc[c1, c2]
            if pd.isna(r):
                continue
            pairs.append((c1, c2, float(r)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    sentences = []
    for c1, c2, r in pairs[:top_n]:
        arah = 'positif' if r > 0 else ('negatif' if r < 0 else 'tidak ada')
        sentences.append(
            f"Korelasi antara {c1} dan {c2} adalah {_fmt_num(r)} "
            f"({_corr_strength_label(r)}, arah {arah})."
        )
    return sentences


def _corr_strength_label(r: float) -> str:
    a = abs(r)
    if a >= 0.8:
        return 'sangat kuat'
    if a >= 0.6:
        return 'kuat'
    if a >= 0.4:
        return 'sedang'
    if a >= 0.2:
        return 'lemah'
    return 'sangat lemah'


def categorical_sentence(df: pd.DataFrame, types: Dict[str, str], column: str) -> Optional[str]:
    """Ringkasan satu kolom kategorikal: jumlah nilai unik + nilai terbanyak."""
    if types.get(column) != 'categorical':
        return None
    vc = df[column].value_counts(dropna=True)
    if vc.empty:
        return None
    total = int(df[column].notna().sum())
    top_val, top_count = vc.index[0], int(vc.iloc[0])
    pct = round(top_count / total * 100, 1) if total else 0
    return (
        f"Kolom {column} memiliki {int(df[column].nunique(dropna=True))} nilai unik. "
        f"Nilai paling sering muncul adalah \"{top_val}\" sebanyak {top_count} kali ({pct}% dari data)."
    )


# ══════════════════════════════════════════════════════════════════════════
# 5. Susun entri Knowledge Base dari CSV
# ══════════════════════════════════════════════════════════════════════════
def build_kb_entries_from_csv(df: pd.DataFrame, types: Dict[str, str], source: str = 'csv') -> List[Dict]:
    """
    Analisis DataFrame CSV lengkap (describe, correlation, statistik lain)
    dan susun jadi daftar entri KB {keywords, answer, source} — bentuk yang
    sama seperti nlp_engine.parse_text_to_entries() — supaya bisa langsung
    disimpan ke SLM aktif dan otomatis terjawab lewat /api/chat + decision
    tree matching (nlp_engine.query_kb) yang sudah ada, tanpa perlu logika
    NLP baru.
    """
    dt_col        = _primary_datetime_col(types)
    numeric_cols     = [c for c, t in types.items() if t == 'numeric']
    categorical_cols = [c for c, t in types.items() if t == 'categorical']

    entries: List[Dict] = []

    # a) Ringkasan umum dataset
    entries.append({
        'keywords': ['ringkasan', 'data', 'dataset', 'statistik', 'gambaran', 'overview', 'summary'],
        'answer':   dataset_overview_sentence(df, types, source_name=source),
    })

    # b) describe() per kolom numerik
    for col in numeric_cols:
        sent = describe_sentence(df, types, col)
        if sent:
            entries.append({
                'keywords': list(dict.fromkeys([col, 'statistik', 'deskripsi', 'describe', 'sebaran'])),
                'answer':   sent,
            })

    # c) Nilai tertinggi / terendah / rata-rata / total per kolom numerik
    AGG_SYNONYMS = [
        ('max',  ['tertinggi', 'maksimum', 'terbesar', 'puncak']),
        ('min',  ['terendah', 'minimum', 'terkecil']),
        ('mean', ['rata-rata', 'ratarata', 'mean']),
        ('sum',  ['total', 'jumlah']),
    ]
    for col in numeric_cols:
        for agg, extra_kw in AGG_SYNONYMS:
            sent = extreme_value_sentence(df, types, col, agg=agg, datetime_col=dt_col)
            if sent:
                entries.append({
                    'keywords': list(dict.fromkeys([col] + extra_kw)),
                    'answer':   sent,
                })

    # d) Korelasi antar kolom numerik
    corr_sents = correlation_sentences(df, types)
    if corr_sents:
        entries.append({
            'keywords': ['korelasi', 'hubungan', 'korelasi antar kolom', 'relasi', 'keterkaitan'],
            'answer':   ' '.join(corr_sents),
        })

    # e) Statistik kolom kategorikal
    for col in categorical_cols:
        sent = categorical_sentence(df, types, col)
        if sent:
            entries.append({
                'keywords': list(dict.fromkeys([col, 'kategori', 'terbanyak', 'unik'])),
                'answer':   sent,
            })

    return [e for e in entries if e.get('answer')]


# ══════════════════════════════════════════════════════════════════════════
# 6. Entry point utama
# ══════════════════════════════════════════════════════════════════════════
def analyze_csv(file_like_or_path, source: str = 'data.csv') -> Dict:
    """
    Fungsi utama: baca CSV (baris pertama = nama kolom), deteksi tipe kolom
    (numerik / tanggal-waktu / kategorikal), hitung describe(), correlation
    matrix, dan statistik kategorikal, lalu hasilkan entri Knowledge Base
    berbahasa Indonesia siap pakai. Mengembalikan dict lengkap untuk
    keperluan API (preview maupun penyimpanan).
    """
    df = load_csv(file_like_or_path)
    if df.empty or len(df.columns) == 0:
        raise ValueError('CSV tidak berisi data yang bisa dibaca')

    types = detect_column_types(df)
    df = coerce_datetime_columns(df, types)

    describe_df = compute_describe(df, types)
    corr_df     = compute_correlation(df, types)
    cat_stats   = compute_categorical_stats(df, types)
    kb_entries  = build_kb_entries_from_csv(df, types, source=source)

    return {
        'source':            source,
        'n_rows':            int(df.shape[0]),
        'n_cols':            int(df.shape[1]),
        'columns':           list(df.columns),
        'column_types':      types,
        'describe':          json.loads(describe_df.reset_index(names='stat').to_json(orient='records')) if not describe_df.empty else [],
        'correlation':       json.loads(corr_df.reset_index(names='kolom').to_json(orient='records')) if not corr_df.empty else [],
        'categorical_stats': cat_stats,
        'kb_entries':        kb_entries,
        'summary_sentence':  dataset_overview_sentence(df, types, source_name=source),
    }
