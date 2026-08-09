"""
tree_renderer.py — Render knowledge base sebagai Decision Tree visual (PNG via matplotlib)

Struktur pohon:
  ROOT
  └─ SOURCE-1
     ├─ Entry (keywords → answer snippet)
     └─ ...
  └─ SOURCE-2
     └─ ...
"""

import io
import math
import textwrap
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from typing import List, Dict

# ── Palette ──────────────────────────────────────────────────────────────────
BG        = '#0d1117'
SURF      = '#161b22'
SURF2     = '#21262d'
BORDER    = '#30363d'
ACCENT    = '#58a6ff'
ACCENT2   = '#7c5cfc'
GREEN     = '#3fb950'
ORANGE    = '#d29922'
MUTED     = '#7d8590'
TEXT      = '#e6edf3'
TEXT_DIM  = '#c9d1d9'
EDGE_CLR  = '#30363d'

SOURCE_COLORS = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149',
    '#7c5cfc', '#39d353', '#ffa657', '#ff7b72',
    '#a5d6ff', '#56d364',
]

# ── Layout constants ──────────────────────────────────────────────────────────
NODE_W      = 3.6   # entry node width (inches)
NODE_H      = 1.05  # entry node height
H_GAP       = 0.45  # horizontal gap between nodes
V_GAP       = 0.70  # vertical gap between rows
SRC_H       = 0.55  # source header height
ROOT_H      = 0.60
ROOT_W      = 2.8
COLS        = 4     # max columns per source group
MARGIN      = 0.55
FONT_MAIN   = 'DejaVu Sans'

# ASCII symbols (avoids missing-glyph warnings on headless servers)
ICON_ROOT = '[BOT]'
ICON_SRC  = '[DOC]'
ICON_KW   = '[KW] '


def _wrap(text: str, width: int) -> str:
    lines = textwrap.wrap(text, width)
    return '\n'.join(lines[:3]) + ('…' if len(lines) > 3 else '')


def _short_answer(ans: str, chars: int = 72) -> str:
    ans = ans.replace('\n', ' ')
    return ans[:chars] + '…' if len(ans) > chars else ans


def build_tree_png(kb: List[Dict], dpi: int = 130) -> bytes:
    """
    Render the full KB decision tree as a PNG and return raw bytes.
    """
    if not kb:
        return _empty_png(dpi)

    # Group entries by source
    groups: Dict[str, List[Dict]] = {}
    for e in kb:
        src = e.get('source', 'manual')
        groups.setdefault(src, []).append(e)

    sources = list(groups.keys())
    n_sources = len(sources)

    # ── Compute canvas size ───────────────────────────────────────────────
    # Each source group: header row + N entry rows
    group_heights = []
    group_cols    = []
    for src in sources:
        n = len(groups[src])
        cols = min(n, COLS)
        rows = math.ceil(n / cols)
        group_heights.append(SRC_H + rows * (NODE_H + V_GAP))
        group_cols.append(cols)

    max_cols    = max(group_cols)
    group_w     = max_cols * (NODE_W + H_GAP) - H_GAP

    total_h = ROOT_H + V_GAP * 0.8 + sum(group_heights) + (n_sources - 1) * V_GAP * 0.6
    total_w = group_w + 2 * MARGIN

    fig_w = max(total_w, 6.0)
    fig_h = max(total_h + 2 * MARGIN, 4.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    # Invert y so we draw top-to-bottom
    ax.invert_yaxis()
    ax.set_ylim(fig_h, 0)

    cx = fig_w / 2   # center x
    y  = MARGIN

    # ── ROOT node ────────────────────────────────────────────────────────
    _draw_box(ax,
              cx - ROOT_W / 2, y,
              ROOT_W, ROOT_H,
              label=f'CS Bot  |  {len(kb)} entries',
              facecolor=ACCENT2, textcolor='#ffffff',
              fontsize=9.5, bold=True, radius=0.12)
    root_cx = cx
    root_by = y + ROOT_H   # bottom y of root

    y += ROOT_H + V_GAP * 0.8

    # ── Source groups ─────────────────────────────────────────────────────
    for g_idx, src in enumerate(sources):
        entries  = groups[src]
        s_color  = SOURCE_COLORS[g_idx % len(SOURCE_COLORS)]
        n        = len(entries)
        cols     = min(n, COLS)
        rows     = math.ceil(n / cols)

        total_row_w = cols * NODE_W + (cols - 1) * H_GAP
        src_x       = cx - total_row_w / 2

        # Source header box
        src_label = f'[SRC]  {src}  ({n} entries)'
        _draw_box(ax,
                  src_x, y,
                  total_row_w, SRC_H,
                  label=src_label,
                  facecolor=SURF2, textcolor=s_color,
                  fontsize=8.2, bold=True, radius=0.09,
                  edge_color=s_color, edge_width=1.4)

        src_cx = src_x + total_row_w / 2
        src_ty = y   # top of source box

        # Edge: root → source
        _draw_edge(ax, root_cx, root_by, src_cx, src_ty, color=s_color, lw=1.3)

        y += SRC_H + V_GAP * 0.45

        # Entry nodes
        for i, entry in enumerate(entries):
            row = i // cols
            col = i %  cols

            ex = src_x + col * (NODE_W + H_GAP)
            ey = y + row * (NODE_H + V_GAP)

            kw_text  = '  ·  '.join(entry['keywords'][:5])
            if len(entry['keywords']) > 5:
                kw_text += f'  +{len(entry["keywords"])-5}'
            ans_text = _short_answer(entry['answer'], 68)

            # Edge: source → entry
            entry_cx = ex + NODE_W / 2
            _draw_edge(ax,
                       src_cx, src_ty + SRC_H,
                       entry_cx, ey,
                       color=s_color, lw=0.8, alpha=0.55)

            _draw_entry_node(ax, ex, ey, NODE_W, NODE_H,
                             kw_text, ans_text, s_color)

        y += rows * (NODE_H + V_GAP) + V_GAP * 0.6

    # ── Footer ────────────────────────────────────────────────────────────
    ax.text(cx, fig_h - 0.18,
            'CS Bot — Decision Tree SLM  ·  Universitas Mataram',
            ha='center', va='bottom', fontsize=7,
            color=MUTED, fontfamily=FONT_MAIN)

    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_box(ax, x, y, w, h, label,
              facecolor=SURF, textcolor=TEXT,
              fontsize=8.5, bold=False, radius=0.08,
              edge_color=BORDER, edge_width=0.8):
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=facecolor,
        edgecolor=edge_color,
        linewidth=edge_width,
        zorder=2,
    )
    ax.add_patch(fancy)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold' if bold else 'normal',
            color=textcolor,
            fontfamily=FONT_MAIN,
            clip_on=True,
            zorder=3)


def _draw_entry_node(ax, x, y, w, h, kw_text, ans_text, accent):
    # Background
    fancy = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0,rounding_size=0.08',
        facecolor=SURF,
        edgecolor=BORDER,
        linewidth=0.7,
        zorder=2,
    )
    ax.add_patch(fancy)

    # Left accent bar
    bar = FancyBboxPatch(
        (x, y), 0.055, h,
        boxstyle='round,pad=0,rounding_size=0.04',
        facecolor=accent,
        edgecolor='none',
        zorder=3,
    )
    ax.add_patch(bar)

    pad = 0.13
    # Keyword line
    ax.text(x + 0.13, y + 0.27,
            f'[KW] {kw_text}',
            ha='left', va='center',
            fontsize=6.5, fontweight='bold',
            color=accent,
            fontfamily=FONT_MAIN,
            clip_on=True, zorder=4)

    # Separator line
    ax.plot([x + 0.10, x + w - pad],
            [y + h * 0.47, y + h * 0.47],
            color=BORDER, lw=0.5, zorder=3)

    # Answer snippet
    ax.text(x + 0.13, y + h * 0.72,
            ans_text,
            ha='left', va='center',
            fontsize=5.8,
            color=TEXT_DIM,
            fontfamily=FONT_MAIN,
            clip_on=True, zorder=4,
            wrap=True)


def _draw_edge(ax, x1, y1, x2, y2, color=EDGE_CLR, lw=1.0, alpha=0.8):
    """Draw a curved edge between two points."""
    mid_y = (y1 + y2) / 2
    ax.annotate('',
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=lw, alpha=alpha,
                    connectionstyle='arc3,rad=0.0',
                ),
                zorder=1)


def _empty_png(dpi: int) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis('off')
    ax.text(0.5, 0.5, 'Knowledge base kosong.\nTambah entry terlebih dahulu.',
            ha='center', va='center', color=MUTED, fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
