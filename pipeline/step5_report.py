"""
step5_report.py
───────────────
Lit benchmark_results.json (Step 4) et génère :
  - 6 figures PNG (scores, radar, pred vs actual, confusion/résidus,
                    erreurs, score vs temps)
  - report.html   : rapport complet navigable dans le navigateur

Usage :
    python step5_report.py
    python step5_report.py --results benchmark_results.json
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # pas de fenêtre GUI — compatible serveur/pipeline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FOLDER = os.path.basename(SCRIPT_DIR).lower()
SUBFOLDER_NAMES = ('pipeline', 'scripts', 'src', 'code', 'pipeline_cnc')
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR) if SCRIPT_FOLDER in SUBFOLDER_NAMES else SCRIPT_DIR

plt.rcParams.update({
    'figure.dpi'       : 120,
    'font.size'        : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

MODEL_COLORS = [
    '#1565C0','#E65100','#2E7D32','#6A1B9A','#C62828',
    '#00838F','#F57F17','#4E342E','#37474F','#880E4F',
]


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    for base in (PROJECT_ROOT, SCRIPT_DIR):
        c = os.path.join(base, path)
        if os.path.exists(c):
            return c
    return path


def short(name):
    name = str(name)
    return name.split('_', 1)[1] if '_' in name else name


def color_map(keys):
    return {short(k): MODEL_COLORS[i % len(MODEL_COLORS)]
            for i, k in enumerate(keys)}


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — Comparaison des scores (barh)
# ─────────────────────────────────────────────────────────────────────────────
def fig_scores(df_m, task, meta, out_dir):
    if task == 'classification':
        plots = [
            ('f1_macro',          'f1_macro_std',   'F1 Macro (CV mean)'),
            ('balanced_accuracy', None,              'Balanced Accuracy'),
            ('accuracy',          'accuracy_std',    'Accuracy'),
        ]
    else:
        plots = [
            ('r2',   'r2_std', 'R² (CV mean)'),
            ('mae',  None,     'MAE'),
            ('rmse', None,     'RMSE'),
        ]

    keys   = df_m['model'] if 'model' in df_m.columns else df_m.index
    cm     = color_map(keys)
    n      = len(plots)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 7))
    if n == 1: axes = [axes]

    for ax, (metric, err_col, label) in zip(axes, plots):
        if metric not in df_m.columns:
            continue
        asc = metric in ('mae', 'rmse')
        df_s = df_m.sort_values(metric, ascending=asc)
        cols = [cm.get(r, '#888') for r in df_s['model']]
        bars = ax.barh(df_s['model'], df_s[metric], color=cols,
                       alpha=0.85, edgecolor='white', linewidth=0.5)

        if err_col and err_col in df_s.columns:
            ax.barh(df_s['model'], df_s[err_col],
                    left=df_s[metric] - df_s[err_col],
                    color='black', alpha=0.2, height=0.4)

        for bar, val in zip(bars, df_s[metric]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)

        best = df_s[metric].min() if asc else df_s[metric].max()
        for bar, val in zip(bars, df_s[metric]):
            if val == best:
                bar.set_edgecolor('gold'); bar.set_linewidth(2.5)

        ax.set_title(label, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Comparaison des modèles — {meta["dataset"]} | {task.upper()}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig1_scores.png')
    plt.savefig(p, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'  ✅ fig1_scores.png')
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — Score vs Temps
# ─────────────────────────────────────────────────────────────────────────────
def fig_score_vs_time(df_m, task, out_dir):
    metric = 'f1_macro' if task == 'classification' else 'r2'
    label  = 'F1 Macro' if task == 'classification' else 'R²'
    cm     = color_map(df_m['model'])

    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in df_m.iterrows():
        if metric not in row or 'fit_time_s' not in row:
            continue
        c = cm.get(row['model'], '#888')
        ax.scatter(row['fit_time_s'], row[metric], color=c, s=130,
                   zorder=5, edgecolors='white', linewidth=1.2)
        ax.annotate(row['model'],
                    (row['fit_time_s'], row[metric]),
                    textcoords='offset points', xytext=(7, 4),
                    fontsize=8, color=c)

    ax.set_xlabel('Temps CV total (s)', fontweight='bold')
    ax.set_ylabel(f'{label} (CV mean)', fontweight='bold')
    ax.set_title(f'{label} vs Temps d\'entraînement', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig2_score_vs_time.png')
    plt.savefig(p, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'  ✅ fig2_score_vs_time.png')
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — Predicted vs Actual
# ─────────────────────────────────────────────────────────────────────────────
def fig_pred_vs_actual(preds_raw, task, out_dir):
    from sklearn.metrics import r2_score, mean_absolute_error
    valid = {k: v for k, v in preds_raw.items() if 'y_true' in v}
    if not valid:
        return None
    n = len(valid)
    nc = min(3, n); nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 5*nr))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    cm = color_map(list(valid.keys()))

    for i, (key, pred) in enumerate(valid.items()):
        ax   = axes_flat[i]
        name = short(key)
        c    = cm.get(name, '#888')
        yt   = np.array(pred['y_true'])
        yp   = np.array(pred['y_pred'])

        if task == 'regression':
            ax.scatter(yt, yp, alpha=0.4, s=20, color=c,
                       edgecolors='white', linewidth=0.3)
            lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            ax.plot(lims, lims, 'k--', linewidth=1.2)
            r2  = r2_score(yt.astype(float), yp.astype(float))
            mae = mean_absolute_error(yt.astype(float), yp.astype(float))
            ax.set_title(f'{name}\nR²={r2:.4f}  MAE={mae:.4f}',
                         fontweight='bold', fontsize=9)
            ax.set_xlabel('Réel'); ax.set_ylabel('Prédit')
        else:
            jitter = np.random.uniform(-0.2, 0.2, len(yt))
            uv = sorted(set(yt) | set(yp))
            v2n = {v: j for j, v in enumerate(uv)}
            try:
                yt_n = np.array([float(v) for v in yt]) + jitter
                yp_n = np.array([float(v) for v in yp]) + jitter
            except Exception:
                yt_n = np.array([v2n[v] for v in yt], float) + jitter
                yp_n = np.array([v2n[v] for v in yp], float) + jitter
            ax.scatter(yt_n, yp_n, alpha=0.35, s=18, color=c)
            acc = np.mean(np.array(yt) == np.array(yp))
            ax.set_title(f'{name}\nAccuracy={acc:.4f}',
                         fontweight='bold', fontsize=9)
            ax.set_xlabel('Réel'); ax.set_ylabel('Prédit')
        ax.grid(True, alpha=0.25)

    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle('Prédictions vs Réel (OOF)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig3_pred_vs_actual.png')
    plt.savefig(p, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'  ✅ fig3_pred_vs_actual.png')
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Radar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar(df_m, task, out_dir):
    if task == 'classification':
        rm = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
        rl = ['Accuracy', 'Bal. Acc.', 'F1 Macro', 'F1 Weighted']
    else:
        for col in ('mae', 'rmse'):
            mn, mx = df_m[col].min(), df_m[col].max()
            df_m[f'{col}_score'] = 1 - (df_m[col] - mn) / (mx - mn + 1e-9)
        df_m['r2_norm'] = (df_m['r2'] - df_m['r2'].min()) / \
                           (df_m['r2'].max() - df_m['r2'].min() + 1e-9)
        rm = ['r2_norm', 'mae_score', 'rmse_score']
        rl = ['R² (norm)', 'MAE score', 'RMSE score']

    rm = [c for c in rm if c in df_m.columns]
    rl = rl[:len(rm)]
    if not rm:
        return None

    N      = len(rm)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    cm     = color_map(df_m['model'])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for _, row in df_m.iterrows():
        vals  = [row.get(m, 0) for m in rm] + [row.get(rm[0], 0)]
        c     = cm.get(row['model'], '#888')
        ax.plot(angles, vals, color=c, linewidth=2, label=row['model'])
        ax.fill(angles, vals, color=c, alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rl, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Radar — Profil multi-métriques', fontweight='bold',
                 fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig4_radar.png')
    plt.savefig(p, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'  ✅ fig4_radar.png')
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5 — Confusion matrices / Résidus
# ─────────────────────────────────────────────────────────────────────────────
def fig_confusion_or_residuals(preds_raw, task, out_dir):
    from sklearn.metrics import confusion_matrix
    valid = {k: v for k, v in preds_raw.items() if 'y_true' in v}
    if not valid:
        return None
    n = len(valid)
    nc = min(3, n); nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 5*nr))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    cm_color  = color_map(list(valid.keys()))

    for i, (key, pred) in enumerate(valid.items()):
        ax   = axes_flat[i]
        name = short(key)
        yt   = np.array(pred['y_true'])
        yp   = np.array(pred['y_pred'])

        if task == 'classification':
            classes = pred.get('classes', sorted(set(yt)))
            cm_val  = confusion_matrix(yt, yp, labels=classes)
            sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=classes, yticklabels=classes,
                        cbar=False, linewidths=0.5)
            acc = np.mean(yt == yp)
            ax.set_title(f'{name}\nAcc={acc:.4f}', fontweight='bold', fontsize=9)
            ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
        else:
            c       = cm_color.get(name, '#888')
            resid   = yt.astype(float) - yp.astype(float)
            ax.scatter(yp.astype(float), resid, alpha=0.4, s=20, color=c)
            ax.axhline(0, color='black', linestyle='--', linewidth=1.2)
            rmse = np.sqrt(np.mean(resid**2))
            ax.set_title(f'{name}\nRMSE={rmse:.4f}', fontweight='bold', fontsize=9)
            ax.set_xlabel('Prédit'); ax.set_ylabel('Résidu')
            ax.grid(True, alpha=0.25)

    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    title = 'Matrices de Confusion' if task == 'classification' else 'Graphes de Résidus'
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig5_confusion_residuals.png')
    plt.savefig(p, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'  ✅ fig5_confusion_residuals.png')
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 6 — Distribution des erreurs (boxplot)
# ─────────────────────────────────────────────────────────────────────────────
def fig_error_dist(preds_raw, task, out_dir):
    valid = {k: v for k, v in preds_raw.items() if 'y_true' in v}
    if not valid:
        return None
    cm = color_map(list(valid.keys()))
    box_data, box_labels, box_colors = [], [], []

    for key, pred in valid.items():
        name = short(key)
        yt   = np.array(pred['y_true'])
        yp   = np.array(pred['y_pred'])
        errs = np.abs(yt.astype(float) - yp.astype(float)) \
               if task == 'regression' else (yt != yp).astype(float)
        box_data.append(errs)
        box_labels.append(name)
        box_colors.append(cm.get(name, '#888'))

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for j, d in enumerate(box_data):
        ax.scatter(j+1, d.mean(), color='black', s=40, zorder=5, marker='D')

    ylabel = 'Erreur Absolue' if task == 'regression' else 'Taux d\'erreur'
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title('Distribution des erreurs par modèle', fontweight='bold', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p = os.path.join(out_dir, 'fig6_error_dist.png')
    plt.savefig(p, bbox_inches='tight', dpi=130)
    plt.close()
    print(f'  ✅ fig6_error_dist.png')
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  RAPPORT HTML
# ─────────────────────────────────────────────────────────────────────────────
def generate_html(data, df_m, task, figures, out_dir):
    meta   = data['meta']
    medals = {0: '🥇', 1: '🥈', 2: '🥉'}

    main_metric = 'f1_macro' if task == 'classification' else 'r2'
    asc         = False
    ranked      = df_m.sort_values(main_metric, ascending=asc).reset_index(drop=True)

    # Tableau métriques HTML
    if task == 'classification':
        cols_show = ['model', 'accuracy', 'balanced_accuracy',
                     'f1_macro', 'f1_weighted', 'fit_time_s']
    else:
        cols_show = ['model', 'r2', 'mae', 'rmse', 'fit_time_s']
    cols_show = [c for c in cols_show if c in ranked.columns]

    rows_html = ''
    for i, row in ranked.iterrows():
        medal   = medals.get(i, '')
        row_cls = 'gold' if i == 0 else ('silver' if i == 1 else
                  ('bronze' if i == 2 else ''))
        cells   = ''.join(
            f'<td>{medal if c == "model" else ""}'
            f'{row[c] if isinstance(row[c], str) else f"{row[c]:.4f}"}</td>'
            for c in cols_show
        )
        rows_html += f'<tr class="{row_cls}">{cells}</tr>\n'

    headers_html = ''.join(f'<th>{c}</th>' for c in cols_show)

    # Images
    imgs_html = ''
    for p in figures:
        if p and os.path.isfile(p):
            fname = os.path.basename(p)
            imgs_html += f'''
            <div class="fig-card">
                <h3>{fname.replace("_"," ").replace(".png","").title()}</h3>
                <img src="{fname}" alt="{fname}">
            </div>'''

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Benchmark CNC — {meta['dataset']}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#0f1117; color:#e0e0e0; margin:0; padding:20px; }}
  h1   {{ color:#61dafb; border-bottom:2px solid #61dafb; padding-bottom:8px; }}
  h2   {{ color:#90caf9; margin-top:40px; }}
  h3   {{ color:#b0bec5; }}
  .meta-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; margin:20px 0; }}
  .meta-card {{ background:#1e2130; border-radius:8px; padding:14px; border-left:4px solid #61dafb; }}
  .meta-card .label {{ color:#90caf9; font-size:0.8em; text-transform:uppercase; }}
  .meta-card .value {{ font-size:1.1em; font-weight:bold; margin-top:4px; }}
  table  {{ border-collapse:collapse; width:100%; margin-top:16px; }}
  th     {{ background:#1565C0; color:white; padding:10px 14px; text-align:left; }}
  td     {{ padding:9px 14px; border-bottom:1px solid #2a2d3e; }}
  tr.gold   td:first-child {{ background:#f57f17; color:black; font-weight:bold; border-radius:4px; }}
  tr.silver td:first-child {{ background:#546e7a; font-weight:bold; }}
  tr.bronze td:first-child {{ background:#6d4c41; font-weight:bold; }}
  tr:hover {{ background:#1e2130; }}
  .fig-card {{ background:#1e2130; border-radius:10px; padding:16px; margin:16px 0; }}
  .fig-card img {{ width:100%; border-radius:6px; }}
  .badge {{ display:inline-block; background:#1565C0; color:white; padding:3px 10px;
            border-radius:12px; font-size:0.85em; margin:3px; }}
</style>
</head>
<body>
<h1>📊 Benchmark CNC — Rapport de Prédiction d'Usure Outil</h1>

<div class="meta-grid">
  <div class="meta-card"><div class="label">Dataset</div><div class="value">{meta['dataset']}</div></div>
  <div class="meta-card"><div class="label">Tâche</div><div class="value">{task.upper()}</div></div>
  <div class="meta-card"><div class="label">Cible</div><div class="value">{meta['target']}</div></div>
  <div class="meta-card"><div class="label">Échantillons</div><div class="value">{meta['n_samples']}</div></div>
  <div class="meta-card"><div class="label">Features</div><div class="value">{meta['n_features']}</div></div>
  <div class="meta-card"><div class="label">CV</div><div class="value">{meta['cv_folds']}-Fold OOF</div></div>
  <div class="meta-card"><div class="label">Généré le</div><div class="value">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div></div>
</div>

<h2>🏆 Classement des modèles</h2>
<table>
  <thead><tr>{headers_html}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>

<h2>📈 Graphes</h2>
{imgs_html}

</body></html>"""

    p = os.path.join(out_dir, 'report.html')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  ✅ report.html')
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Step 5 — Génération du rapport et des graphes'
    )
    parser.add_argument('--results', type=str,
                        default='benchmark_results.json')
    args = parser.parse_args()

    results_path = resolve_path(args.results)
    if not os.path.isfile(results_path):
        print(f'[ERROR] Fichier manquant : {results_path}')
        print('        Lance d\'abord step4_benchmark.py')
        sys.exit(1)

    with open(results_path, encoding='utf-8') as f:
        data = json.load(f)

    meta      = data['meta']
    task      = meta['task']
    metrics   = data['metrics']
    preds_raw = data['predictions']

    print(f'\n{"═" * 62}')
    print(f'  STEP 5 — GÉNÉRATION DU RAPPORT')
    print(f'  Dataset : {meta["dataset"]}  |  Tâche : {task.upper()}')
    print(f'{"═" * 62}\n')

    # DataFrame métriques
    rows   = [{'model': short(k), **v} for k, v in metrics.items()]
    df_m   = pd.DataFrame(rows)
    df_m   = df_m[df_m['status'] == 'ok'].copy()

    out_dir = SCRIPT_DIR
    figures = []
    print('  Génération des figures...')
    figures.append(fig_scores(df_m.copy(), task, meta, out_dir))
    figures.append(fig_score_vs_time(df_m.copy(), task, out_dir))
    figures.append(fig_pred_vs_actual(preds_raw, task, out_dir))
    figures.append(fig_radar(df_m.copy(), task, out_dir))
    figures.append(fig_confusion_or_residuals(preds_raw, task, out_dir))
    figures.append(fig_error_dist(preds_raw, task, out_dir))

    figures = [f for f in figures if f]

    print('\n  Génération du rapport HTML...')
    html_path = generate_html(data, df_m.copy(), task, figures, out_dir)

    print(f'\n{"═" * 62}')
    print(f'  ✅ PIPELINE COMPLÈTE')
    print(f'  Rapport → {html_path}')
    print(f'  Ouvre report.html dans ton navigateur pour tout voir.')
    print(f'{"═" * 62}\n')


if __name__ == '__main__':
    main()
