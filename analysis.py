#!/usr/bin/env python3
"""Step 2: statistics and figures from data/features.npz.

    python3 analysis.py

Writes figures/*.png, *.svg, figures/frequency_sweep.gif and data/results.json
(every number quoted in the README). About three minutes.

Rule used throughout: the unit of analysis is the CHILD, not the window.
A child contributes one averaged value to every group comparison, and all of
a child's windows stay on the same side of every train/test split.
"""
import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import mne
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
HERE = os.path.dirname(os.path.abspath(__file__))
FIG, DATA = os.path.join(HERE, 'figures'), os.path.join(HERE, 'data')
plt.style.use(os.path.join(FIG, 'neuro.mplstyle'))
INK, MUTED, GREY, BLUE, ORANGE = '#0b0b0b', '#52514e', '#8a8985', '#2a78d6', '#eb6834'
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
RNG = np.random.default_rng(0)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png')); fig.savefig(os.path.join(FIG, name + '.svg'))
    plt.close(fig)


def cohen_d(a, b):
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)


def load():
    d = np.load(os.path.join(DATA, 'features.npz'))
    psd, f, subj, lab, ch = d['psd'], d['freqs'], d['subject'], d['label'], list(d['channels'])
    total = psd[:, :, (f >= 1) & (f < 45)].sum(2)
    rel = np.stack([psd[:, :, (f >= a) & (f < b)].sum(2) / total for a, b in BANDS.values()], 2)
    y = (lab == 'ADHD').astype(int)
    kids = np.unique(subj)
    y_kid = np.array([y[subj == k][0] for k in kids])
    rel_kid = np.stack([rel[subj == k].mean(0) for k in kids])                  # child x ch x band
    spec_kid = np.stack([(psd[subj == k] / total[subj == k][:, :, None]).mean(0) for k in kids])
    return dict(f=f, ch=ch, rel=rel, y=y, subj=subj, y_kid=y_kid, rel_kid=rel_kid, spec_kid=spec_kid)


def info_for(ch):
    info = mne.create_info(ch, 128.0, 'eeg')
    info.set_montage('standard_1020')
    return info


# ── group differences, one value per child ──────────────────────────────────
def permutation_p(values, y, n=5000):
    """Two-sided p for a difference in means, shuffling which child is in which group."""
    obs = values[y == 1].mean(0) - values[y == 0].mean(0)
    count = np.zeros_like(obs)
    for _ in range(n):
        yp = RNG.permutation(y)
        count += np.abs(values[yp == 1].mean(0) - values[yp == 0].mean(0)) >= np.abs(obs)
    return (count + 1) / (n + 1)


def fig_topomaps(D):
    names = ['theta', 'alpha', 'beta']
    maps = {n: D['rel_kid'][:, :, list(BANDS).index(n)] for n in names}
    maps['theta / beta'] = maps['theta'] / maps['beta']
    info = info_for(D['ch'])
    y = D['y_kid']
    pvals = np.stack([permutation_p(v, y) for v in maps.values()])              # 4 x 19
    sig = fdrcorrection(pvals.ravel(), alpha=0.05)[0].reshape(pvals.shape)

    fig, ax = plt.subplots(3, 4, figsize=(9.6, 7.4))
    out = {}
    for j, (name, v) in enumerate(maps.items()):
        a, c = v[y == 1], v[y == 0]
        lim = (min(a.mean(0).min(), c.mean(0).min()), max(a.mean(0).max(), c.mean(0).max()))
        for i, grp in enumerate((a, c)):
            im, _ = mne.viz.plot_topomap(grp.mean(0), info, axes=ax[i, j], show=False, cmap='viridis',
                                         vlim=lim, contours=4, sensors=True)
        cb = fig.colorbar(im, ax=ax[:2, j], orientation='horizontal', fraction=0.045, pad=0.03, aspect=18)
        cb.set_label('ratio' if '/' in name else 'share of 1-45 Hz power', fontsize=8); cb.ax.tick_params(labelsize=7)
        d = cohen_d(a, c)
        mask = sig[j]
        im2, _ = mne.viz.plot_topomap(d, info, axes=ax[2, j], show=False, cmap='RdBu_r', vlim=(-0.9, 0.9),
                                      contours=0, mask=mask,
                                      mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                                                       markersize=6, markeredgewidth=1.2))
        ax[0, j].set_title(name, fontsize=11, fontweight='bold', loc='center')
        out[name] = dict(strongest_channel=D['ch'][int(np.argmax(np.abs(d)))], strongest_d=float(d[np.argmax(np.abs(d))]),
                         n_significant=int(mask.sum()), significant=[D['ch'][k] for k in np.where(mask)[0]])
    cb = fig.colorbar(im2, ax=ax[2, :], orientation='horizontal', fraction=0.05, pad=0.08, aspect=40)
    cb.set_label("Effect size (Cohen's d):  blue = lower in ADHD,  red = higher in ADHD.  "
                 'White dots: significant after correction for 76 tests', fontsize=8.5)
    for i, t in enumerate(('ADHD\n(61 children)', 'Control\n(60 children)', 'Difference')):
        ax[i, 0].text(-0.28, 0.5, t, transform=ax[i, 0].transAxes, ha='center', va='center',
                      rotation=90, fontsize=10, fontweight='bold')
    save(fig, 'topomaps')
    return out


def fig_theta_beta(D):
    cz = D['ch'].index('Cz')
    kid = D['rel_kid']
    tbr = kid[:, cz, 1] / kid[:, cz, 3]
    y = D['y_kid']
    a, c = tbr[y == 1], tbr[y == 0]
    p = mannwhitneyu(a, c).pvalue
    d = float(cohen_d(a, c))
    auc = roc_auc_score(y, tbr)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.28))
    for i, (v, col, name) in enumerate(((c, GREY, f'Control\n({len(c)} children)'), (a, ORANGE, f'ADHD\n({len(a)} children)'))):
        x = i + RNG.uniform(-0.17, 0.17, len(v))
        ax.scatter(x, v, s=30, color=col, edgecolor='#fcfcfb', linewidth=0.8, zorder=3)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.34, np.median(v), f'{np.median(v):.2f}', va='center', fontsize=9)
    ax.set_xticks([0, 1], [f'Control\n({len(c)} children)', f'ADHD\n({len(a)} children)'])
    ax.set_xlim(-0.5, 1.7); ax.set_ylabel('Theta / beta ratio at Cz'); ax.grid(axis='x', visible=False)
    ax.set_title('Theta/beta is lower in ADHD here, not higher')

    f = D['f']; m = (f >= 1) & (f <= 40)
    for grp, col, name in ((0, GREY, 'Control'), (1, ORANGE, 'ADHD')):
        s = D['spec_kid'][y == grp][:, cz, :][:, m] * 100
        boot = np.array([s[RNG.integers(0, len(s), len(s))].mean(0) for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax2.fill_between(f[m], lo, hi, color=col, alpha=0.2, linewidth=0)
        ax2.plot(f[m], s.mean(0), color=col)
        k = np.searchsorted(f[m], 2.5 if grp == 0 else 17)
        ax2.text(f[m][k] + (0.6 if grp == 0 else 0), s.mean(0)[k] * (1.25 if grp == 0 else 1.6), name, color=INK, fontsize=9)
    for name, (lo_, hi_) in list(BANDS.items())[:4]:
        ax2.axvline(hi_, color='#e6e5e0', linewidth=0.8)
        ax2.text((lo_ + min(hi_, 40)) / 2, 0.985, name, transform=ax2.get_xaxis_transform(), ha='center', va='top',
                 fontsize=8, color=MUTED)
    ax2.set_yscale('log'); ax2.set_xlim(1, 40)
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Share of power per 0.5 Hz at Cz (%)')
    ax2.set_title('Group average spectrum (band = 95% interval across children)')
    ax2.grid(axis='x', visible=False)
    save(fig, 'theta_beta')
    return dict(median_adhd=float(np.median(a)), median_control=float(np.median(c)), cohen_d=d,
                mann_whitney_p=float(p), auc_of_ratio_alone=float(auc))


# ── the honest-split test ───────────────────────────────────────────────────
def fig_split(D, n_perm=100):
    X = np.log(D['rel']).reshape(len(D['y']), -1)                 # 19 channels x 5 bands per window
    y, g = D['y'], D['subj']
    models = {'Logistic regression': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              'Random forest': lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1)}
    res = {}
    for name, make in models.items():
        by_window, by_child = [], []
        for seed in range(5):
            by_window.append(cross_val_score(make(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean())
            by_child.append(cross_val_score(make(), X, y, groups=g,
                                            cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed)).mean())
        res[name] = dict(by_window=float(np.mean(by_window)), by_child=float(np.mean(by_child)),
                         by_child_sd=float(np.std(by_child)), by_child_runs=[float(v) for v in by_child])
    # chance level: shuffle which CHILD carries which label, then repeat the child-wise test
    kids = np.unique(g); y_kid = np.array([y[g == k][0] for k in kids])
    null = []
    for i in range(n_perm):
        lab = dict(zip(kids, RNG.permutation(y_kid)))
        yp = np.array([lab[k] for k in g])
        null.append(cross_val_score(models['Logistic regression'](), X, yp, groups=g,
                                    cv=StratifiedGroupKFold(5, shuffle=True, random_state=i)).mean())
    null = np.array(null)
    lr = res['Logistic regression']['by_child']
    res['chance'] = dict(mean=float(null.mean()), p95=float(np.percentile(null, 95)),
                         p_value_logistic=float(((null >= lr).sum() + 1) / (n_perm + 1)), n_permutations=n_perm)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.axhspan(100 * np.percentile(null, 2.5), 100 * np.percentile(null, 97.5), color='#eeeeea', zorder=0)
    ax.text(1.46, 100 * null.mean(), 'chance\n(labels shuffled)', color=MUTED, fontsize=8.5, va='center', ha='right')
    for i, (name, r) in enumerate(list(res.items())[:2]):
        ax.plot([0, 1], [100 * r['by_window'], 100 * r['by_child']], color=[BLUE, ORANGE][i], marker='o', markersize=8,
                markeredgecolor='#fcfcfb')
        ax.text(-0.06, 100 * r['by_window'], f"{name}  {100 * r['by_window']:.0f}%", ha='right', va='center', fontsize=9)
        top = r['by_child'] == max(res[k]['by_child'] for k in models)
        ax.text(1.06, 100 * r['by_child'] + (1.2 if top else -1.2), f"{100 * r['by_child']:.0f}%", ha='left', va='center', fontsize=9)
    ax.set_xticks([0, 1], ['Windows shuffled\n(same child in train and test)', 'Whole children held out\n(the honest test)'])
    ax.set_xlim(-0.95, 1.5); ax.set_ylim(40, 90); ax.set_ylabel('Accuracy (%)')
    ax.set_title('The same classifier, tested two ways'); ax.grid(axis='x', visible=False)
    save(fig, 'honest_split')
    return res


# ── moving figure: scalp map swept across frequency ─────────────────────────
def gif_sweep(D):
    f, y, info = D['f'], D['y_kid'], info_for(D['ch'])
    centres = np.arange(2.0, 30.5, 0.5)
    frames = []
    for c in centres:
        m = (f >= c - 1) & (f <= c + 1)
        v = D['spec_kid'][:, :, m].sum(2)
        frames.append((v[y == 1].mean(0), v[y == 0].mean(0), cohen_d(v[y == 1], v[y == 0])))
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 3.3))
    title = fig.suptitle('', x=0.03, ha='left', fontsize=11, fontweight='bold')

    def draw(k):
        a, c, d = frames[k]
        lim = (min(a.min(), c.min()), max(a.max(), c.max()))
        for axis, v, name in ((ax[0], a, 'ADHD'), (ax[1], c, 'Control')):
            axis.clear(); mne.viz.plot_topomap(v, info, axes=axis, show=False, cmap='viridis', vlim=lim, contours=4)
            axis.set_title(name, loc='center', fontsize=10)
        ax[2].clear(); mne.viz.plot_topomap(d, info, axes=ax[2], show=False, cmap='RdBu_r', vlim=(-0.9, 0.9), contours=0)
        ax[2].set_title('Difference (red = higher in ADHD)', loc='center', fontsize=10)
        band = next(n for n, (lo, hi) in BANDS.items() if lo <= centres[k] < hi)
        title.set_text(f'Where the power is at {centres[k]:4.1f} Hz  ({band})')

    out = os.path.join(FIG, 'frequency_sweep.gif')
    FuncAnimation(fig, draw, frames=len(frames)).save(out, writer=PillowWriter(fps=6), dpi=90)
    plt.close(fig)
    return os.path.getsize(out) / 1e6


def main():
    D = load()
    res = dict(children=int(len(D['y_kid'])), adhd=int(D['y_kid'].sum()), control=int((1 - D['y_kid']).sum()),
               windows=int(len(D['y'])))
    res['theta_beta_at_Cz'] = fig_theta_beta(D)
    res['scalp_maps'] = fig_topomaps(D)
    res['classifier'] = fig_split(D)
    res['gif_MB'] = round(gif_sweep(D), 2)
    json.dump(res, open(os.path.join(DATA, 'results.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
