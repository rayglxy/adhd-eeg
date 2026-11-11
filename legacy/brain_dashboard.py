#!/usr/bin/env python3
"""
ADHD vs Neurotypical — Full Brain EEG Dashboard
Symmetric, ergonomic layout.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import mne
import pandas as pd
from scipy import signal as sci_signal

# ── Config ───────────────────────────────────────────────────────
DATA_PATH = '/Users/test/Desktop/eeg_visualizer/adhdata.csv'
SFREQ     = 128
WINDOW_S  = 6

CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']

BANDS = [
    ('Delta', 1,  4),
    ('Theta', 4,  8),
    ('Alpha', 8,  12),
    ('Beta',  13, 30),
    ('Gamma', 30, 45),
]
BAND_COLS = ['#4a90d9','#b07ee8','#27ae60','#e67e22','#e74c3c']

ADHD_COL  = '#ff6b6b'
NT_COL    = '#6bffb8'
BG        = '#08080e'
PANEL     = '#0f0f18'
GRID_COL  = '#1a1a28'
BORDER    = '#22223a'
TEXT_DIM  = '#556677'
TEXT_MID  = '#8899aa'
TEXT_BRT  = '#bbccdd'

# ── Load data ────────────────────────────────────────────────────
print("Loading EEG data ...")
df = pd.read_csv(DATA_PATH)

def pick_subject(df, cls, min_s=WINDOW_S * SFREQ + 200):
    for sid in df[df['Class'] == cls]['ID'].unique():
        sub = df[df['ID'] == sid][CH_NAMES].values.T.astype(float)
        if sub.shape[1] >= min_s:
            return sub, sid
    raise RuntimeError(f"No subject for {cls}")

adhd_raw, adhd_id = pick_subject(df, 'ADHD')
ctrl_raw, ctrl_id = pick_subject(df, 'Control')
WIN = WINDOW_S * SFREQ
offset = min(adhd_raw.shape[1], ctrl_raw.shape[1]) // 3
adhd_win = adhd_raw[:, offset:offset + WIN]
ctrl_win = ctrl_raw[:, offset:offset + WIN]

print("Computing group statistics ...")
def group_band_powers(df, cls):
    powers = {b[0]: [] for b in BANDS}
    for sid in df[df['Class'] == cls]['ID'].unique()[:20]:
        sub = df[df['ID'] == sid][CH_NAMES].values.T.astype(float)
        if sub.shape[1] < WINDOW_S * SFREQ:
            continue
        chunk = sub[:, :WINDOW_S * SFREQ]
        for name, lo, hi in BANDS:
            freqs, psd = sci_signal.welch(chunk, fs=SFREQ, nperseg=256)
            idx = (freqs >= lo) & (freqs <= hi)
            powers[name].append(psd[:, idx].mean())
    return {k: (np.mean(v), np.std(v)) for k, v in powers.items()}

adhd_group = group_band_powers(df, 'ADHD')
ctrl_group = group_band_powers(df, 'Control')

# ── MNE ──────────────────────────────────────────────────────────
mne.set_log_level('WARNING')
info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
info.set_montage('standard_1020')
pos_2d = np.array([info['chs'][i]['loc'][:2] for i in range(19)])
pos_norm = pos_2d.copy()
pos_norm[:, 0] /= (np.abs(pos_norm[:, 0]).max() + 0.01)
pos_norm[:, 1] /= (np.abs(pos_norm[:, 1]).max() + 0.01)
pos_norm *= 0.78

# ── DSP ──────────────────────────────────────────────────────────
def band_power_ch(data_2d, fmin, fmax):
    freqs, psd = sci_signal.welch(data_2d, fs=SFREQ, nperseg=256)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return psd[:, idx].mean(axis=1)

def theta_beta_ratio(data_2d):
    return band_power_ch(data_2d, 4, 8) / (band_power_ch(data_2d, 13, 30) + 1e-12)

def psd_avg(data_2d):
    freqs, psd = sci_signal.welch(data_2d, fs=SFREQ, nperseg=256)
    return freqs, psd.mean(axis=0)

def connectivity(data_2d):
    corr = np.corrcoef(data_2d)
    np.fill_diagonal(corr, 0)
    return corr

# ── Reusable panel styling ───────────────────────────────────────
def style_ax(ax, title=None):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT_DIM, labelsize=7)
    if title:
        ax.set_title(title, color=TEXT_BRT, fontsize=9.5,
                     fontweight='bold', pad=8, fontfamily='monospace')

def add_panel_border(ax):
    """Rounded border around info panels."""
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle='round,pad=0.015', linewidth=1,
        edgecolor=BORDER, facecolor='none')
    ax.add_patch(rect)

# ── Drawing functions ────────────────────────────────────────────
def draw_head(ax, color='#333348'):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta) * 0.92, np.sin(theta) * 0.92,
            color=color, linewidth=1.6, alpha=0.5)
    ax.plot([-0.07, 0, 0.07], [0.92, 1.05, 0.92],
            color=color, linewidth=1.4, alpha=0.5)
    for s in [-1, 1]:
        t = np.linspace(-0.3, 0.3, 30)
        ax.plot(s * (0.92 + 0.05 * np.cos(t * np.pi)),
                t * 0.65, color=color, linewidth=1.2, alpha=0.4)

def draw_brain(ax, data, color, title):
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.15, 1.25)
    ax.set_aspect('equal')
    ax.set_facecolor(PANEL)
    ax.axis('off')

    draw_head(ax)

    corr = connectivity(data)
    lines, alphas = [], []
    for i in range(19):
        for j in range(i + 1, 19):
            if abs(corr[i, j]) > 0.50:
                lines.append([pos_norm[i], pos_norm[j]])
                alphas.append(abs(corr[i, j]))
    if lines:
        lc = LineCollection(lines, linewidths=0.7, alpha=0.25)
        lc.set_color([(*matplotlib.colors.to_rgb(color), a * 0.5) for a in alphas])
        ax.add_collection(lc)

    tbr = theta_beta_ratio(data)
    tbr_n = (tbr - tbr.min()) / (tbr.max() - tbr.min() + 1e-12)

    # Triple-layer glow
    for size_mult, alpha in [(700, 0.08), (350, 0.20), (140, 0.85)]:
        ax.scatter(pos_norm[:, 0], pos_norm[:, 1],
                   s=tbr_n * size_mult + size_mult * 0.2,
                   c=[color] * 19, alpha=alpha, edgecolors='none',
                   zorder=3 if alpha > 0.5 else 2)

    # White core dots
    ax.scatter(pos_norm[:, 0], pos_norm[:, 1],
               s=tbr_n * 25 + 12, c='white', alpha=0.7,
               edgecolors='none', zorder=4)

    for k, name in enumerate(CH_NAMES):
        ax.annotate(name, pos_norm[k], textcoords='offset points',
                    xytext=(7, 5), fontsize=5, color=TEXT_DIM,
                    fontfamily='monospace', alpha=0.7)

    ax.set_title(title, color=color, fontsize=12, fontweight='bold',
                 pad=10, fontfamily='monospace')

def draw_psd(ax, adhd_data, ctrl_data):
    f_a, p_a = psd_avg(adhd_data)
    f_c, p_c = psd_avg(ctrl_data)
    p_a_db = 10 * np.log10(p_a + 1e-12)
    p_c_db = 10 * np.log10(p_c + 1e-12)
    mask = f_a <= 50

    for (name, lo, hi), col in zip(BANDS, BAND_COLS):
        ax.axvspan(lo, hi, alpha=0.06, color=col)

    ax.plot(f_a[mask], p_a_db[mask], color=ADHD_COL, linewidth=1.4, alpha=0.9, label='ADHD')
    ax.fill_between(f_a[mask], p_a_db[mask], p_a_db[mask].min(), alpha=0.05, color=ADHD_COL)
    ax.plot(f_c[mask], p_c_db[mask], color=NT_COL, linewidth=1.4, alpha=0.9, label='Neurotypical')
    ax.fill_between(f_c[mask], p_c_db[mask], p_c_db[mask].min(), alpha=0.05, color=NT_COL)

    # Band labels at top
    ymax = max(p_a_db[mask].max(), p_c_db[mask].max()) + 1.5
    for (name, lo, hi), col in zip(BANDS, BAND_COLS):
        ax.text((lo + hi) / 2, ymax, name, ha='center', va='bottom',
                fontsize=6.5, color=col, fontweight='bold', alpha=0.8)

    style_ax(ax, 'Power Spectral Density')
    ax.set_xlabel('Frequency (Hz)', color=TEXT_DIM, fontsize=7.5)
    ax.set_ylabel('Power (dB)', color=TEXT_DIM, fontsize=7.5)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.2, edgecolor=BORDER)
    ax.grid(True, color=GRID_COL, linewidth=0.3, alpha=0.6)

def draw_radar(ax, adhd_data, ctrl_data):
    labels = [b[0] for b in BANDS]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    def vals(data):
        v = [band_power_ch(data, lo, hi).mean() for _, lo, hi in BANDS]
        v = np.array(v)
        return (v / (v.max() + 1e-12)).tolist()

    a_v = vals(adhd_data) + vals(adhd_data)[:1]
    c_v = vals(ctrl_data) + vals(ctrl_data)[:1]

    ax.set_facecolor(PANEL)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, color=TEXT_MID, fontsize=8)

    ax.plot(angles, a_v, color=ADHD_COL, linewidth=1.6, alpha=0.85, label='ADHD')
    ax.fill(angles, a_v, color=ADHD_COL, alpha=0.10)
    ax.plot(angles, c_v, color=NT_COL, linewidth=1.6, alpha=0.85, label='Neurotypical')
    ax.fill(angles, c_v, color=NT_COL, alpha=0.10)

    ax.set_ylim(0, 1.15)
    ax.set_title('Band Power Profile', color=TEXT_BRT, fontsize=9.5,
                 fontweight='bold', pad=18, fontfamily='monospace')
    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.2, edgecolor=BORDER)
    ax.tick_params(colors=TEXT_DIM, labelsize=6)
    ax.spines['polar'].set_color(BORDER)
    ax.grid(color=GRID_COL, linewidth=0.3, alpha=0.6)

def draw_group_bars(ax, a_grp, c_grp):
    labels = [b[0] for b in BANDS]
    x = np.arange(len(labels))
    w = 0.3

    a_m = np.array([a_grp[b][0] for b in labels])
    c_m = np.array([c_grp[b][0] for b in labels])
    a_s = np.array([a_grp[b][1] for b in labels])
    c_s = np.array([c_grp[b][1] for b in labels])
    mx = max(a_m.max(), c_m.max())
    a_m, c_m, a_s, c_s = a_m/mx, c_m/mx, a_s/mx, c_s/mx

    ax.bar(x - w/2, a_m, w, color=ADHD_COL, alpha=0.70, yerr=a_s,
           capsize=3, error_kw={'linewidth':0.7, 'color':'#ff9999'}, label='ADHD (n=20)')
    ax.bar(x + w/2, c_m, w, color=NT_COL, alpha=0.70, yerr=c_s,
           capsize=3, error_kw={'linewidth':0.7, 'color':'#99ffcc'}, label='Control (n=20)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Norm. Power', color=TEXT_DIM, fontsize=7.5)
    style_ax(ax, 'Group-Level Band Power (mean \u00b1 SD)')
    ax.legend(fontsize=6.5, framealpha=0.2, edgecolor=BORDER)
    ax.grid(True, axis='y', color=GRID_COL, linewidth=0.3, alpha=0.5)

def draw_dataset(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(PANEL); ax.axis('off')
    add_panel_border(ax)

    items = [
        (0.50, 0.88, "DATASET OVERVIEW", '#ffffff', 10, 'bold'),
        (0.50, 0.74, "Kaggle ADHD EEG Dataset", TEXT_MID, 8.5, 'normal'),
        (0.50, 0.65, "19 channels  \u00b7  10\u201320 system  \u00b7  128 Hz", TEXT_DIM, 7.5, 'normal'),
        (0.28, 0.50, "61", ADHD_COL, 16, 'bold'),
        (0.28, 0.40, "ADHD", ADHD_COL, 7.5, 'normal'),
        (0.72, 0.50, "60", NT_COL, 16, 'bold'),
        (0.72, 0.40, "Control", NT_COL, 7.5, 'normal'),
        (0.50, 0.25, "2,166,383", '#ffffff', 13, 'bold'),
        (0.50, 0.16, "total EEG samples", TEXT_DIM, 7.5, 'normal'),
    ]
    for x, y, text, color, size, weight in items:
        ax.text(x, y, text, ha='center', va='center', color=color,
                fontsize=size, fontweight=weight, fontfamily='monospace')

    # Divider line between counts
    ax.plot([0.50, 0.50], [0.36, 0.58], color=BORDER, linewidth=1, alpha=0.5)

def draw_metrics(ax, adhd_data, ctrl_data):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(PANEL); ax.axis('off')
    add_panel_border(ax)

    tbr_a = theta_beta_ratio(adhd_data).mean()
    tbr_c = theta_beta_ratio(ctrl_data).mean()
    theta_a = band_power_ch(adhd_data, 4, 8).mean()
    theta_c = band_power_ch(ctrl_data, 4, 8).mean()
    beta_a  = band_power_ch(adhd_data, 13, 30).mean()
    beta_c  = band_power_ch(ctrl_data, 13, 30).mean()
    alpha_a = band_power_ch(adhd_data, 8, 12).mean()
    alpha_c = band_power_ch(ctrl_data, 8, 12).mean()

    ax.text(0.50, 0.90, "KEY METRICS", ha='center', va='center',
            color='#ffffff', fontsize=10, fontweight='bold', fontfamily='monospace')

    metrics = [
        ("\u03b8/\u03b2 Ratio",     tbr_a,   tbr_c,   ".2f"),
        ("Theta (\u03bcV\u00b2)",   theta_a, theta_c, ".0f"),
        ("Beta (\u03bcV\u00b2)",    beta_a,  beta_c,  ".0f"),
        ("Alpha (\u03bcV\u00b2)",   alpha_a, alpha_c, ".0f"),
    ]

    y = 0.76
    bar_w = 0.28
    for label, va, vc, fmt in metrics:
        ax.text(0.50, y, label, ha='center', va='center',
                color=TEXT_MID, fontsize=7.5, fontfamily='monospace')
        y -= 0.045
        # Values
        ax.text(0.12, y, format(va, fmt), ha='center', va='center',
                color=ADHD_COL, fontsize=8, fontweight='bold', fontfamily='monospace')
        ax.text(0.88, y, format(vc, fmt), ha='center', va='center',
                color=NT_COL, fontsize=8, fontweight='bold', fontfamily='monospace')
        # Bars
        mx = max(va, vc) + 1e-12
        aw = (va / mx) * bar_w
        cw = (vc / mx) * bar_w
        bh = 0.022
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.50 - aw - 0.005, y - bh/2), aw, bh,
            boxstyle='round,pad=0.003', facecolor=ADHD_COL, alpha=0.45))
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.505, y - bh/2), cw, bh,
            boxstyle='round,pad=0.003', facecolor=NT_COL, alpha=0.45))
        y -= 0.105

# ══════════════════════════════════════════════════════════════════
#  SYMMETRIC LAYOUT — 4 rows × 2 columns (all equal width)
# ══════════════════════════════════════════════════════════════════
print("Rendering dashboard ...")
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 11), facecolor=BG, dpi=180)

margin = 0.04
gs = gridspec.GridSpec(
    4, 2, figure=fig,
    top=0.92, bottom=margin, left=margin + 0.01, right=1 - margin - 0.01,
    hspace=0.38, wspace=0.14,
    height_ratios=[1.6, 1.0, 1.0, 0.85],
)

# Row 0 — brains
ax_bl = fig.add_subplot(gs[0, 0])
ax_br = fig.add_subplot(gs[0, 1])
# Row 1 — PSD / Radar
ax_psd   = fig.add_subplot(gs[1, 0])
ax_radar = fig.add_subplot(gs[1, 1], projection='polar')
# Row 2 — group bars (left) / group bars mirrored (right)
# Actually: group bars left, waveform comparison right
ax_group = fig.add_subplot(gs[2, 0])
ax_wave  = fig.add_subplot(gs[2, 1])
# Row 3 — dataset / metrics
ax_dset = fig.add_subplot(gs[3, 0])
ax_metr = fig.add_subplot(gs[3, 1])

# ── Title ────────────────────────────────────────────────────────
fig.text(0.50, 0.98,
         'ADHD vs Neurotypical Brain  \u2014  EEG Analysis Dashboard',
         ha='center', va='top', color='white',
         fontsize=17, fontweight='bold', fontfamily='monospace')
fig.text(0.50, 0.948,
         'Real EEG data  \u00b7  121 subjects  \u00b7  2.1M samples  \u00b7  19 channels  \u00b7  10\u201320 montage',
         ha='center', va='top', color=TEXT_DIM, fontsize=8.5, fontfamily='monospace')

# Center divider
fig.add_artist(plt.Line2D([0.50, 0.50], [margin, 0.93],
               transform=fig.transFigure, color=BORDER,
               linewidth=0.8, alpha=0.5))

# ── Render panels ────────────────────────────────────────────────
draw_brain(ax_bl, adhd_win, ADHD_COL, 'ADHD Brain Activity')
draw_brain(ax_br, ctrl_win, NT_COL, 'Neurotypical Brain Activity')
draw_psd(ax_psd, adhd_win, ctrl_win)
draw_radar(ax_radar, adhd_win, ctrl_win)
draw_group_bars(ax_group, adhd_group, ctrl_group)

# Waveform comparison (symmetric counterpart to group bars)
FZ = CH_NAMES.index('Fz')
n_wave = 3 * SFREQ
t = np.linspace(0, 3, n_wave)
ax_wave.plot(t, adhd_win[FZ, :n_wave], color=ADHD_COL, linewidth=0.8,
             alpha=0.85, label='ADHD')
ax_wave.plot(t, ctrl_win[FZ, :n_wave], color=NT_COL, linewidth=0.8,
             alpha=0.85, label='Neurotypical')
ax_wave.axhline(0, color=GRID_COL, linewidth=0.5)
style_ax(ax_wave, 'Fz Waveform Comparison')
ax_wave.set_xlabel('Time (s)', color=TEXT_DIM, fontsize=7.5)
ax_wave.set_ylabel('\u03bcV', color=TEXT_DIM, fontsize=7.5)
ax_wave.legend(fontsize=6.5, framealpha=0.2, edgecolor=BORDER)
ax_wave.grid(True, color=GRID_COL, linewidth=0.3, alpha=0.5)

draw_dataset(ax_dset)
draw_metrics(ax_metr, adhd_win, ctrl_win)

# ── Save ─────────────────────────────────────────────────────────
out = '/Users/test/Desktop/eeg_visualizer/eeg_brain_dashboard.png'
fig.savefig(out, facecolor=BG, dpi=180)
print(f"\nSaved: {out}")
plt.close()
