#!/usr/bin/env python3
"""
ADHD vs Neurotypical EEG Dashboard
Real-data brain visualization — LinkedIn content
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import mne
import pandas as pd
from scipy import signal as sci_signal

# ── Config ───────────────────────────────────────────────────────
DATA_PATH   = '/Users/test/Desktop/eeg_visualizer/adhdata.csv'
SFREQ       = 128          # Hz (standard for this dataset)
WINDOW_S    = 4            # seconds used per FFT analysis window
WAVE_S      = 3            # seconds shown in waveform panel
INTERVAL_MS = 650          # ms between animation frames
STEP        = 32           # samples to advance per frame (~0.25 s)

CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']
FZ_IDX = CH_NAMES.index('Fz')   # 16

BANDS = [
    ('Delta', 1,  4,  '#4a90d9'),
    ('Theta', 4,  8,  '#b07ee8'),
    ('Alpha', 8,  12, '#27ae60'),
    ('Beta',  13, 30, '#e67e22'),
    ('Gamma', 30, 45, '#e74c3c'),
]

ADHD_COL = '#ff6b6b'
NT_COL   = '#6bffb8'
BG       = '#0d0d0d'
PANEL    = '#131313'

# ── Load & select one subject per class ──────────────────────────
print("Loading EEG data ...")
df = pd.read_csv(DATA_PATH)

def pick_subject(df, cls, min_s=WINDOW_S * SFREQ + 200):
    for sid in df[df['Class'] == cls]['ID'].unique():
        sub = df[df['ID'] == sid][CH_NAMES].values.T.astype(float)
        if sub.shape[1] >= min_s:
            return sub, sid
    raise RuntimeError(f"No qualifying subject for class '{cls}'")

adhd_raw, adhd_id = pick_subject(df, 'ADHD')
ctrl_raw, ctrl_id = pick_subject(df, 'Control')
print(f"  ADHD    -> {adhd_id}  ({adhd_raw.shape[1]} samples)")
print(f"  Control -> {ctrl_id}  ({ctrl_raw.shape[1]} samples)")
WIN = WINDOW_S * SFREQ
MAX_OFFSET = min(adhd_raw.shape[1], ctrl_raw.shape[1]) - WIN - 1

# ── MNE montage ──────────────────────────────────────────────────
mne.set_log_level('WARNING')
info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
info.set_montage('standard_1020')

# ── DSP helpers ──────────────────────────────────────────────────
def band_power(data_2d, fmin, fmax):
    """Mean band power per channel; data_2d: (n_ch, n_samp)."""
    freqs, psd = sci_signal.welch(data_2d, fs=SFREQ,
                                   nperseg=min(256, data_2d.shape[1]))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return psd[:, idx].mean(axis=1)          # (n_ch,)

def theta_beta_ratio(data_2d):
    theta = band_power(data_2d, 4, 8)
    beta  = band_power(data_2d, 13, 30)
    return theta / (beta + 1e-12)

def all_band_powers(data_2d):
    return {name: band_power(data_2d, lo, hi).mean()
            for name, lo, hi, _ in BANDS}

def state_text(tbr_mean, bp):
    theta_dom = bp['Theta'] > bp['Alpha']
    alpha_dom = bp['Alpha'] > bp['Theta']
    if tbr_mean > 3.5:
        return "Elevated Theta  \u00b7  Attention Dysregulation"
    elif tbr_mean > 2.2 and theta_dom:
        return "Theta Dominant  \u00b7  Reduced Inhibitory Control"
    elif alpha_dom and bp['Beta'] > bp['Alpha'] * 0.4:
        return "Alpha-Beta Balance  \u00b7  Focused Cognition"
    elif alpha_dom:
        return "Alpha Dominant  \u00b7  Relaxed Awareness"
    else:
        return "Mixed Band Activity  \u00b7  Active Processing"

# ── Figure ───────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9), facecolor=BG, dpi=100)

gs = gridspec.GridSpec(4, 2, figure=fig,
                       top=0.91, bottom=0.04,
                       left=0.05, right=0.97,
                       hspace=0.60, wspace=0.28,
                       height_ratios=[2.2, 1.4, 1.4, 0.8])

ax_topo = [fig.add_subplot(gs[0, c]) for c in range(2)]
ax_bars = [fig.add_subplot(gs[1, c]) for c in range(2)]
ax_wave = [fig.add_subplot(gs[2, c]) for c in range(2)]
ax_lbl  = [fig.add_subplot(gs[3, c]) for c in range(2)]

for ax in ax_topo + ax_bars + ax_wave + ax_lbl:
    ax.set_facecolor(PANEL)

# Global text (drawn once)
fig.text(0.5, 0.975,
         'ADHD vs Neurotypical Brain  \u2014  Live EEG Dashboard',
         ha='center', va='top', color='white',
         fontsize=16, fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.942,
         'Real EEG dataset  \u00b7  Theta/Beta Ratio  \u00b7  Band Power Spectra  \u00b7  Fz Waveform',
         ha='center', va='top', color='#666666',
         fontsize=9, fontfamily='monospace')
fig.text(0.275, 0.921, '\u25cf  ADHD', ha='center',
         color=ADHD_COL, fontsize=12, fontweight='bold')
fig.text(0.725, 0.921, '\u25cf  Neurotypical', ha='center',
         color=NT_COL, fontsize=12, fontweight='bold')

fig.add_artist(plt.Line2D([0.5, 0.5], [0.04, 0.92],
               transform=fig.transFigure,
               color='#2a2a2a', linewidth=1.2))

# ── Animation ────────────────────────────────────────────────────
def update(frame):
    offset = (frame * STEP) % MAX_OFFSET

    datasets = [
        (adhd_raw[:, offset:offset + WIN], ADHD_COL),
        (ctrl_raw[:, offset:offset + WIN], NT_COL),
    ]

    for col_i, (data, color) in enumerate(datasets):
        tbr = theta_beta_ratio(data)
        bp  = all_band_powers(data)

        # Topomap
        ax = ax_topo[col_i]
        ax.cla(); ax.set_facecolor(PANEL)
        vmax = max(float(tbr.max()), 0.1)
        mne.viz.plot_topomap(
            tbr, info, axes=ax, show=False,
            cmap='RdYlBu_r', vlim=(0, vmax),
            sensors=True, contours=4,
        )
        ax.set_title('Theta / Beta Ratio', color='#888888', fontsize=8.5, pad=3)

        # Band power bars
        ax = ax_bars[col_i]
        ax.cla(); ax.set_facecolor(PANEL)
        names    = [b[0] for b in BANDS]
        values   = np.array([bp[n] for n in names])
        bcolors  = [b[3] for b in BANDS]
        normed   = values / (values.max() + 1e-12)

        bars = ax.barh(names, normed, color=bcolors, height=0.55, alpha=0.88)
        for b, c in zip(bars, bcolors):
            ax.barh(b.get_y() + b.get_height() / 2,
                    b.get_width(), height=0.12,
                    color='white', alpha=0.18, left=0)

        ax.set_xlim(0, 1.12)
        ax.tick_params(colors='#888888', labelsize=8.5)
        ax.set_title('Frequency Band Power', color='#888888', fontsize=8.5)
        ax.set_xlabel('Relative power', color='#555555', fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a2a')

        # Waveform
        ax = ax_wave[col_i]
        ax.cla(); ax.set_facecolor(PANEL)
        n_wave = WAVE_S * SFREQ
        fz = data[FZ_IDX, :n_wave]
        t  = np.linspace(0, WAVE_S, n_wave)
        ax.plot(t, fz, color=color, linewidth=0.75, alpha=0.92)
        ax.fill_between(t, fz, alpha=0.08, color=color)
        ax.axhline(0, color='#2d2d2d', linewidth=0.6)
        ax.set_xlabel('Time (s)', color='#555555', fontsize=7.5)
        ax.set_ylabel('\u03bcV', color='#555555', fontsize=7.5)
        ax.set_title('Fz  (Frontal Midline)', color='#888888', fontsize=8.5)
        ax.tick_params(colors='#888888', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a2a')

        # State label
        ax = ax_lbl[col_i]
        ax.cla(); ax.set_facecolor(PANEL)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        label = state_text(tbr.mean(), bp)
        ax.text(0.5, 0.62, label,
                ha='center', va='center',
                color=color, fontsize=9.5, fontweight='bold',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.45',
                          facecolor='#1c1c1c',
                          edgecolor=color, linewidth=1.6, alpha=0.95))
        ax.text(0.5, 0.12,
                f'TBR: {tbr.mean():.2f}  \u00b7  \u03b8: {bp["Theta"]:.1f}  \u03b2: {bp["Beta"]:.1f}',
                ha='center', va='center', color='#555555', fontsize=7.5)

ani = animation.FuncAnimation(
    fig, update,
    interval=INTERVAL_MS,
    cache_frame_data=False,
)

plt.show()
