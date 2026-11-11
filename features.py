#!/usr/bin/env python3
"""Step 1: raw CSV -> one clean table of band powers per 4-second window.

    python3 features.py

Reads adhdata.csv (not in the repo; see README for the download) and writes
data/features.npz. Every later script reads that file, so the 267 MB CSV is
only parsed once.

Processing, per child:
  1. remove each channel's mean (the recording has a constant offset)
  2. band-pass 1-45 Hz, zero-phase FIR (keeps timing undistorted)
  3. re-reference to the average of all 19 channels
  4. cut into non-overlapping 4 s windows
  5. drop windows whose peak-to-peak amplitude is more than 3x that child's
     median window (blinks, movement)
  6. Welch power spectrum per window and channel
"""
import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'adhdata.csv')
OUT = os.path.join(HERE, 'data', 'features.npz')

SFREQ = 128.0
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
WINDOW_S = 4.0
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
REJECT_FACTOR = 3.0


def main():
    mne.set_log_level('ERROR')
    df = pd.read_csv(CSV)
    win = int(WINDOW_S * SFREQ)
    psd_all, subj, label, kept_frac = [], [], [], {}
    for sid, g in df.groupby('ID', sort=True):
        x = g[CHANNELS].to_numpy(float).T                       # channels x samples
        x -= x.mean(axis=1, keepdims=True)
        x = mne.filter.filter_data(x, SFREQ, 1.0, 45.0, method='fir', phase='zero')
        x -= x.mean(axis=0, keepdims=True)                      # average reference
        n = x.shape[1] // win
        ep = x[:, :n * win].reshape(len(CHANNELS), n, win).transpose(1, 0, 2)   # windows x ch x samples
        p2p = np.ptp(ep, axis=2).max(axis=1)
        keep = p2p <= REJECT_FACTOR * np.median(p2p)
        kept_frac[sid] = keep.mean()
        f, psd = welch(ep[keep], fs=SFREQ, nperseg=256, noverlap=128, axis=2)   # 0.5 Hz resolution
        psd_all.append(psd.astype(np.float32))
        subj += [sid] * int(keep.sum())
        label += [g['Class'].iloc[0]] * int(keep.sum())
    psd = np.concatenate(psd_all)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, psd=psd, freqs=f, subject=np.array(subj), label=np.array(label),
                        channels=np.array(CHANNELS))
    k = np.array(list(kept_frac.values()))
    print(f'{len(kept_frac)} children, {len(subj)} clean windows '
          f'({100 * k.mean():.0f}% of windows kept on average, worst child {100 * k.min():.0f}%)')
    print('saved', OUT, f'{os.path.getsize(OUT) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
