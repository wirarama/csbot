"""
scraper.py — Web scraping tipis untuk CS Bot.

Mengambil HTML dari sebuah URL lalu mengekstrak teks dari elemen yang cocok
dengan HTML tag dan/atau CSS class tertentu (atau CSS selector bebas untuk
kasus lanjutan). Setiap elemen yang cocok diperlakukan sebagai satu
paragraf, sama seperti satu paragraf pada upload dokumen .txt — sehingga
bisa langsung masuk ke pipeline NLP (nlp_engine.parse_text_to_entries) yang
sudah ada.
"""

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    'Mozilla/5.0 (compatible; CSBotScraper/1.0; +https://localhost)'
)


def build_selector(tag: str = None, css_class: str = None, selector: str = None) -> str:
    """
    Bangun CSS selector dari kombinasi input:
      - selector diisi        -> dipakai apa adanya (mode lanjutan, override tag/class)
      - tag + css_class diisi -> 'tag.class'
      - hanya tag             -> 'tag'
      - hanya css_class       -> '.class'
    Return None kalau semuanya kosong.
    """
    selector = (selector or '').strip()
    if selector:
        return selector

    tag = (tag or '').strip()
    css_class = (css_class or '').strip().lstrip('.')

    if tag and css_class:
        return f'{tag}.{css_class}'
    if tag:
        return tag
    if css_class:
        return f'.{css_class}'
    return None


def fetch_html(url: str, timeout: int = 12) -> str:
    resp = requests.get(url, timeout=timeout, headers={'User-Agent': USER_AGENT})
    resp.raise_for_status()
    return resp.text


def extract_paragraphs(html: str, css_selector: str) -> list:
    """Ambil teks bersih dari setiap elemen yang cocok dengan css_selector."""
    soup = BeautifulSoup(html, 'html.parser')
    try:
        nodes = soup.select(css_selector)
    except Exception as e:
        raise ValueError(f'CSS selector tidak valid: {e}')

    paragraphs = []
    for node in nodes:
        text = node.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())
        if text:
            paragraphs.append(text)
    return paragraphs


def scrape(url: str, tag: str = None, css_class: str = None,
           selector: str = None, timeout: int = 12):
    """
    Ambil halaman `url`, ekstrak teks dari elemen yang cocok.
    Return (paragraphs: List[str], used_selector: str).
    Raises ValueError untuk input/selector tidak valid,
    requests.exceptions.RequestException untuk masalah jaringan/HTTP.
    """
    css_selector = build_selector(tag, css_class, selector)
    if not css_selector:
        raise ValueError('Isi minimal salah satu: HTML tag, CSS class, atau selector CSS lanjutan')

    html = fetch_html(url, timeout=timeout)
    paragraphs = extract_paragraphs(html, css_selector)
    if not paragraphs:
        raise ValueError(f'Tidak ada elemen yang cocok dengan selector "{css_selector}"')

    return paragraphs, css_selector
