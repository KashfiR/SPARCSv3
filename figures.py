"""Rebuild every figure and the results table from results.json.

    python figures.py --results results/results.json --outdir results/figures

Every point is a mean over replicate seeds and every error bar is the
confidence interval recorded by experiments.py. Pass --errorbar std to draw one
standard deviation instead. Nothing here is hardcoded.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')

METHOD_ORDER = ['SPARCS', 'Spearman-ISIS', 'CPSS-Spearman', 'ASGDR', 'ASGDC']
COLOR = {'SPARCS': '#e15759', 'Spearman-ISIS': '#4e79a7', 'CPSS-Spearman': '#59a14f',
         'ASGDR': '#f28e2b', 'ASGDC': '#f28e2b'}
LABELS = {'s1': 'Synthetic 20% Nonlinear (s1)', 's2': 'Synthetic 80% Nonlinear (s2)',
          'liver': 'Binary Classification (Liver HCC)',
          'leukemia': 'Multiclass Classification (Leukemia)'}

ERRORBAR = 'ci'


def methods_of(entry):
    return [m for m in METHOD_ORDER if m in entry['methods']]


def stat(entry, method, metric):
    """The aggregated record for one metric, or None if it was never measured."""
    return (entry['methods'].get(method, {}).get('aggregate') or {}).get(metric)


def mean_of(entry, method, metric, default=0.0):
    record = stat(entry, method, metric)
    return default if record is None else record['mean']


def error_of(entry, method, metric):
    """Half-height of the error bar, as a (lower, upper) pair."""
    record = stat(entry, method, metric)
    if record is None or record.get('n', 1) < 2:
        return 0.0, 0.0
    if ERRORBAR == 'std':
        return record['std'], record['std']
    return (max(record['mean'] - record['ci_low'], 0.0),
            max(record['ci_high'] - record['mean'], 0.0))


def errors_for(entry, methods, metric):
    pairs = [error_of(entry, m, metric) for m in methods]
    return np.array([[lo for lo, _ in pairs], [hi for _, hi in pairs]])


def headline_metric(task):
    return {'regression': 'test_r2', 'binary': 'test_auc'}.get(task, 'test_f1_macro')


def score_label(task):
    return {'regression': 'Test $R^2$', 'binary': 'Test AUC'}.get(task, 'Test F1-Macro')


def n_replicates(datasets):
    counts = {m.get('n_replicates', 1)
              for entry in datasets.values() for m in entry['methods'].values()}
    return max(counts) if counts else 1


def caption_suffix(datasets):
    n = n_replicates(datasets)
    if n < 2:
        return 'single run'
    return (f"mean of {n} seeds, error bars show the "
            f"{'standard deviation' if ERRORBAR == 'std' else '95% confidence interval'}")


def new_axes(keys, width=7.0, height=4.6):
    rows = int(np.ceil(len(keys) / 2))
    cols = 1 if len(keys) == 1 else 2
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(keys):]:
        ax.axis('off')
    return fig, flat


def fig_feature_recovery(datasets, outdir):
    keys = [k for k, v in datasets.items()
            if any(stat(v, m, 'tpr') for m in methods_of(v))]
    if not keys:
        return
    fig, axes = new_axes(keys)

    for ax, key in zip(axes, keys):
        entry = datasets[key]
        methods = methods_of(entry)
        x = np.arange(len(methods))
        ax.bar(x - 0.2, [mean_of(entry, m, 'tpr') for m in methods], 0.4,
               yerr=errors_for(entry, methods, 'tpr'), capsize=4,
               label='TPR', color='#e15759', alpha=0.9,
               error_kw={'ecolor': '#333333', 'lw': 1.2})
        ax.bar(x + 0.2, [mean_of(entry, m, 'fdr') for m in methods], 0.4,
               yerr=errors_for(entry, methods, 'fdr'), capsize=4,
               label='FDR', color='#edc948', alpha=0.9, hatch='//',
               error_kw={'ecolor': '#333333', 'lw': 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylabel('Rate', fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_title(LABELS.get(key, key), fontweight='bold')
        ax.legend()

    fig.suptitle(f"Feature recovery ({caption_suffix(datasets)})", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / 'fig1_feature_recovery.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_performance_runtime(datasets, outdir):
    keys = list(datasets)
    fig, axes = new_axes(keys)

    for ax, key in zip(axes, keys):
        entry = datasets[key]
        metric = headline_metric(entry['task'])
        for method in methods_of(entry):
            if stat(entry, method, metric) is None:
                continue
            x = mean_of(entry, method, 'runtime_min')
            y = mean_of(entry, method, metric)
            xerr = np.array(error_of(entry, method, 'runtime_min')).reshape(2, 1)
            yerr = np.array(error_of(entry, method, metric)).reshape(2, 1)
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=11,
                        color=COLOR[method], ecolor='#555555', elinewidth=1.2,
                        capsize=3, markeredgecolor='black', markeredgewidth=1.2,
                        alpha=0.92, zorder=3)
            ax.annotate(method, (x, y), fontsize=8,
                        textcoords='offset points', xytext=(9, 7))
        if entry['task'] == 'regression':
            ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
        ax.set_xlabel('Runtime (minutes)', fontweight='bold')
        ax.set_ylabel(score_label(entry['task']), fontweight='bold')
        ax.set_title(LABELS.get(key, key), fontweight='bold')

    fig.suptitle(f"Performance against runtime ({caption_suffix(datasets)})",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / 'fig2_performance_vs_runtime.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_jaccard(datasets, outdir):
    keys = list(datasets)
    fig, axes = new_axes(keys, width=6.2, height=5.2)

    for ax, key in zip(axes, keys):
        entry = datasets[key]
        methods = methods_of(entry)
        pairs = (entry.get('similarity') or {}).get('jaccard', {})

        matrix = np.eye(len(methods))
        annot = [['' for _ in methods] for _ in methods]
        for i, m1 in enumerate(methods):
            annot[i][i] = '1.00'
            for j, m2 in enumerate(methods):
                if i == j:
                    continue
                record = pairs.get(f"{m1}|{m2}") or pairs.get(f"{m2}|{m1}")
                if record:
                    matrix[i, j] = record['mean']
                    annot[i][j] = (f"{record['mean']:.2f}\n$\\pm${record['std']:.2f}"
                                   if record.get('n', 1) > 1 else f"{record['mean']:.2f}")

        sns.heatmap(matrix, annot=np.array(annot), fmt='', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=methods, yticklabels=methods, ax=ax,
                    annot_kws={'fontsize': 8}, cbar_kws={'label': 'Jaccard index'})
        ax.set_title(LABELS.get(key, key), fontweight='bold')
        ax.tick_params(labelsize=8)

    fig.suptitle(f"Cross-method overlap ({caption_suffix(datasets)})", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / 'fig3_jaccard_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_intersections(datasets, outdir):
    keys = list(datasets)
    fig, axes = new_axes(keys, height=5.0)

    for ax, key in zip(axes, keys):
        entry = datasets[key]
        methods = methods_of(entry)
        counts = (entry.get('similarity') or {}).get('intersection', {})

        names, values, lows, highs = [], [], [], []
        for m1, m2 in combinations(methods, 2):
            record = counts.get(f"{m1}|{m2}") or counts.get(f"{m2}|{m1}")
            if not record:
                continue
            names.append(f"{m1}\nvs {m2}")
            values.append(record['mean'])
            if record.get('n', 1) > 1 and ERRORBAR != 'std':
                lows.append(max(record['mean'] - record['ci_low'], 0.0))
                highs.append(max(record['ci_high'] - record['mean'], 0.0))
            else:
                lows.append(record.get('std', 0.0))
                highs.append(record.get('std', 0.0))

        if not names:
            ax.axis('off')
            continue

        bars = ax.bar(names, values, yerr=np.array([lows, highs]), capsize=4,
                      color=sns.color_palette('Set2', len(names)), alpha=0.9,
                      error_kw={'ecolor': '#333333', 'lw': 1.2})
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}",
                    ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Shared features', fontweight='bold')
        ax.set_title(LABELS.get(key, key), fontweight='bold')
        ax.tick_params(axis='x', labelsize=7)

    fig.suptitle(f"Shared features between method pairs ({caption_suffix(datasets)})",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / 'fig4_intersections.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_sparsity_runtime(datasets, outdir):
    keys = list(datasets)
    present = [m for m in METHOD_ORDER if any(m in datasets[k]['methods'] for k in keys)]
    x = np.arange(len(present))
    width = 0.8 / max(len(keys), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for i, key in enumerate(keys):
        entry = datasets[key]
        offset = (i - (len(keys) - 1) / 2) * width
        ax1.bar(x + offset, [mean_of(entry, m, 'n_selected') for m in present], width,
                yerr=errors_for(entry, present, 'n_selected'), capsize=3,
                label=LABELS.get(key, key), alpha=0.9,
                error_kw={'ecolor': '#333333', 'lw': 1.0})
        ax2.bar(x + offset, [max(mean_of(entry, m, 'runtime_min'), 1e-3) for m in present],
                width, yerr=errors_for(entry, present, 'runtime_min'), capsize=3,
                label=LABELS.get(key, key), alpha=0.9,
                error_kw={'ecolor': '#333333', 'lw': 1.0})

    for ax, ylabel, title in ((ax1, 'Features selected', 'Selection sparsity'),
                              (ax2, 'Runtime (minutes)', 'Computational efficiency')):
        ax.set_xticks(x)
        ax.set_xticklabels(present, fontsize=9)
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
    ax2.set_yscale('log')

    fig.suptitle(f"Sparsity and cost ({caption_suffix(datasets)})", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / 'fig5_sparsity_runtime.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_selection_stability(datasets, outdir):
    """Mean pairwise Jaccard between a method's own selections across seeds."""
    keys = [k for k in datasets
            if any(datasets[k]['methods'].get(m, {}).get('selection_stability') is not None
                   for m in methods_of(datasets[k]))]
    if not keys:
        return

    present = [m for m in METHOD_ORDER if any(m in datasets[k]['methods'] for k in keys)]
    x = np.arange(len(present))
    width = 0.8 / max(len(keys), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, key in enumerate(keys):
        entry = datasets[key]
        offset = (i - (len(keys) - 1) / 2) * width
        values = [entry['methods'].get(m, {}).get('selection_stability') or 0.0
                  for m in present]
        bars = ax.bar(x + offset, values, width, label=LABELS.get(key, key), alpha=0.9)
        for bar, value in zip(bars, values):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{value:.2f}", ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=9)
    ax.set_ylabel('Mean pairwise Jaccard across seeds', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Selection stability under reseeding "
                 f"({n_replicates(datasets)} seeds)", fontweight='bold')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / 'fig6_selection_stability.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------

def tex_escape(text):
    """Escape characters that LaTeX would otherwise read as markup.

    The panel labels contain a percent sign, which silently comments out the
    rest of the line if it reaches the .tex file unescaped.
    """
    for char in ('\\', '&', '%', '$', '#', '_', '{', '}'):
        text = text.replace(char, '\\' + char)
    return text.replace('~', '\\textasciitilde{}').replace('^', '\\textasciicircum{}')


def cell(entry, method, metric, digits=3):
    record = stat(entry, method, metric)
    if record is None:
        return '--'
    if record.get('n', 1) < 2:
        return f"{record['mean']:.{digits}f}"
    return f"{record['mean']:.{digits}f} $\\pm$ {record['std']:.{digits}f}"


COLUMNS = {
    'regression': ([('n_selected', '$|S|$', 1), ('test_r2', 'Test $R^2$', 3),
                    ('test_rmse', 'Test RMSE', 2), ('tpr', 'TPR', 3),
                    ('fdr', 'FDR', 3), ('runtime_min', 'Runtime (min)', 2)]),
    'binary': ([('n_selected', '$|S|$', 1), ('test_accuracy', 'Accuracy', 3),
                ('test_f1', 'F1', 3), ('test_auc', 'AUC', 3),
                ('runtime_min', 'Runtime (min)', 2)]),
    'multiclass': ([('n_selected', '$|S|$', 1), ('test_accuracy', 'Accuracy', 3),
                    ('test_f1_macro', 'F1-Macro', 3), ('test_f1_weighted', 'F1-Weighted', 3),
                    ('runtime_min', 'Runtime (min)', 2)]),
}


def latex_table(datasets, outdir):
    panels = []
    for letter, key in zip('ABCD', datasets):
        entry = datasets[key]
        columns = COLUMNS[entry['task']]
        methods = methods_of(entry)

        header = ' & '.join(['Method'] + [title for _, title, _ in columns]) + r' \\'
        rows = []
        for method in methods:
            cells = [cell(entry, method, metric, digits) for metric, _, digits in columns]
            rows.append(' & '.join([tex_escape(method)] + cells) + r' \\')

        stability = [entry['methods'][m].get('selection_stability') for m in methods]
        note = ''
        if any(s is not None for s in stability):
            parts = [f"{tex_escape(m)} {s:.2f}"
                     for m, s in zip(methods, stability) if s is not None]
            note = (f"\\\\[2pt]{{\\footnotesize Selection stability across seeds "
                    f"(mean pairwise Jaccard): {', '.join(parts)}.}}")

        panels.append(
            f"\\textbf{{Panel {letter}: {tex_escape(LABELS.get(key, key))}}}\\\\[2pt]\n"
            f"\\begin{{tabular}}{{l{'r' * len(columns)}}}\n\\toprule\n{header}\n\\midrule\n"
            + "\n".join(rows) + f"\n\\bottomrule\n\\end{{tabular}}{note}\\\\[10pt]\n")

    n = n_replicates(datasets)
    preamble = (f"% generated by figures.py, do not edit by hand\n"
                f"% entries are mean $\\pm$ standard deviation over {n} seeds\n")
    (outdir / 'table1.tex').write_text(preamble + "\n".join(panels))


def main():
    ap = argparse.ArgumentParser(description="Rebuild the paper figures and results table")
    ap.add_argument('--results', default='results/results.json')
    ap.add_argument('--outdir', default='results/figures')
    ap.add_argument('--errorbar', choices=['ci', 'std'], default='ci',
                    help="confidence interval (default) or one standard deviation")
    args = ap.parse_args()

    global ERRORBAR
    ERRORBAR = args.errorbar

    datasets = json.loads(Path(args.results).read_text())['datasets']
    if not datasets:
        raise SystemExit("no datasets found in the results file")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig_feature_recovery(datasets, outdir)
    fig_performance_runtime(datasets, outdir)
    fig_jaccard(datasets, outdir)
    fig_intersections(datasets, outdir)
    fig_sparsity_runtime(datasets, outdir)
    fig_selection_stability(datasets, outdir)
    latex_table(datasets, outdir)

    n = n_replicates(datasets)
    print(f"figures and table1.tex written to {outdir} "
          f"({len(datasets)} dataset(s), {n} seed(s))")
    if n < 2:
        print("only one replicate found, error bars are omitted; "
              "run more seeds with --seeds")


if __name__ == '__main__':
    main()
