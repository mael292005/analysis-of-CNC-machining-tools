"""
plot_timeseries_phm.py
─────────────────────
Visualise une ou deux passes brutes du dataset PHM 2010 (signaux temporels).

Usage :
    # Juste la passe 1
    python plot_timeseries_phm.py --dataset_path ./dataset_4

    # Passe 1 + passe 150 + figure de comparaison côte à côte
    python plot_timeseries_phm.py --dataset_path ./dataset_4 --compare_idx 149

    # Choix libre
    python plot_timeseries_phm.py --dataset_path ./dataset_4 --pass_idx 0 --compare_idx 149 --tool c1
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ────────────────────────────────────────────────────────────
FS        = 50_000
COL_NAMES = ['force_x', 'force_y', 'force_z', 'vib_x', 'vib_y', 'vib_z', 'ae']
WINDOW_MS = 40
COLORS    = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A', '#C62828', '#00838F', '#F57F17']

CHANNEL_LABELS = {
    'force_x': 'Force X (N)',
    'force_y': 'Force Y (N)',
    'force_z': 'Force Z (N)',
    'vib_x'  : 'Vibration X (m/s2)',
    'vib_y'  : 'Vibration Y (m/s2)',
    'vib_z'  : 'Vibration Z (m/s2)',
    'ae'     : 'Acoustic Emission (V)',
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def find_signal_files(dataset_path, tool):
    patterns = [
        os.path.join(dataset_path, tool, '**', '*.csv'),
        os.path.join(dataset_path, '**', f'{tool}_*.csv'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if 'wear' not in os.path.basename(f).lower()]
    return sorted(set(files))


def load_signal(filepath):
    return pd.read_csv(filepath, header=None, names=COL_NAMES)


def load_wear(dataset_path, tool):
    candidates = [
        os.path.join(dataset_path, 'wear', f'{tool}_wear.csv'),
        os.path.join(dataset_path, tool, f'{tool}_wear.csv'),
        os.path.join(dataset_path, f'{tool}_wear.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                df_raw = pd.read_csv(path, header=None)
                df_num = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='all')
                if df_num.shape[1] >= 4:
                    df = df_num.iloc[:, :4].copy()
                    df.columns = ['passe', 'vb1', 'vb2', 'vb3']
                    df['vb_mean'] = df[['vb1', 'vb2', 'vb3']].mean(axis=1)
                elif df_num.shape[1] >= 2:
                    df = df_num.iloc[:, :2].copy()
                    df.columns = ['passe', 'vb_mean']
                else:
                    return None
                df = df.dropna(subset=['passe', 'vb_mean']).reset_index(drop=True)
                df['passe'] = df['passe'].astype(int)
                if df['vb_mean'].max() > 10:
                    df['vb_mean'] = df['vb_mean'] / 1000.0
                return df
            except Exception:
                return None
    return None


def get_vb_label(wear_df, passe_num):
    if wear_df is not None and passe_num in wear_df['passe'].values:
        vb = wear_df.loc[wear_df['passe'] == passe_num, 'vb_mean'].values[0]
        return f'  |  VB = {vb:.3f} mm'
    return ''


def load_pass_data(files, pass_idx):
    pass_idx  = max(0, min(pass_idx, len(files) - 1))
    sig       = load_signal(files[pass_idx])
    passe_num = pass_idx + 1
    return sig, files[pass_idx], passe_num


def style_ax(ax):
    ax.tick_params(colors='white', labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_facecolor('#0d1117')
    ax.grid(True, alpha=0.15, color='white', linewidth=0.5)


# ── Plot individuel (style identique à l'original) ────────────────────────────
def plot_pass(dataset_path, tool, pass_idx):
    files = find_signal_files(dataset_path, tool)
    if not files:
        print(f"[ERROR] Aucun fichier signal pour '{tool}' dans '{dataset_path}'")
        subdirs = [d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))]
        print("        Outils disponibles :", subdirs or "aucun sous-dossier")
        return

    sig, filepath, passe_num = load_pass_data(files, pass_idx)
    wear_df  = load_wear(dataset_path, tool)
    vb_label = get_vb_label(wear_df, passe_num)

    print(f"[INFO] Outil : {tool.upper()} | Passe {passe_num}/{len(files)}")
    print(f"       Fichier : {os.path.basename(filepath)}")

    n_pts  = len(sig)
    dur_ms = n_pts / FS * 1000
    n_show = int(WINDOW_MS * FS / 1000)
    t_ms   = np.arange(n_show) / FS * 1000
    n_ch   = len(COL_NAMES)

    fig = plt.figure(figsize=(14, 2.8 * n_ch), facecolor='#0d1117')
    gs  = gridspec.GridSpec(n_ch, 1, hspace=0.35,
                            top=0.93, bottom=0.05, left=0.08, right=0.98)
    fig.suptitle(
        f'PHM 2010 — Outil {tool.upper()} — Passe {passe_num}/{len(files)}{vb_label}\n'
        f'Fenetre : {WINDOW_MS} ms   |   Duree totale : {dur_ms:.0f} ms   |   '
        f'fs = {FS:,} Hz   |   {n_pts:,} points/canal',
        fontsize=11, color='white', fontweight='bold'
    )

    for i, (col, color) in enumerate(zip(COL_NAMES, COLORS)):
        ax  = fig.add_subplot(gs[i])
        y   = sig[col].values[:n_show]
        rms = np.sqrt(np.mean(y ** 2))
        pk  = y.max() - y.min()

        ax.plot(t_ms, y, color=color, linewidth=0.7, alpha=0.9)
        ax.fill_between(t_ms, y, alpha=0.08, color=color)
        ax.set_ylabel(CHANNEL_LABELS.get(col, col), color='white', fontsize=9)
        style_ax(ax)
        ax.text(0.01, 0.92, f'RMS={rms:.3f}  Pk-Pk={pk:.3f}',
                transform=ax.transAxes, fontsize=7.5, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#222', alpha=0.6))
        if i < n_ch - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Temps (ms)', color='white', fontsize=9)

    out_name = f'timeseries_{tool}_pass{passe_num:03d}.png'
    plt.savefig(out_name, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    print(f"[OK]   Sauvegarde -> {out_name}")
    plt.close(fig)


# ── Figure de comparaison côte à côte ────────────────────────────────────────
def plot_comparison(dataset_path, tool, idx_a, idx_b):
    """
    Génère une figure avec les deux passes côte à côte, canal par canal.
    Gauche = passe A  |  Droite = passe B
    """
    files = find_signal_files(dataset_path, tool)
    if not files:
        print("[ERROR] Aucun fichier signal trouvé.")
        return

    sig_a, _, num_a = load_pass_data(files, idx_a)
    sig_b, _, num_b = load_pass_data(files, idx_b)
    wear_df = load_wear(dataset_path, tool)
    vb_a    = get_vb_label(wear_df, num_a)
    vb_b    = get_vb_label(wear_df, num_b)

    print(f"[INFO] Comparaison : Passe {num_a} vs Passe {num_b} | Outil {tool.upper()}")

    n_show = int(WINDOW_MS * FS / 1000)
    t_ms   = np.arange(n_show) / FS * 1000
    n_ch   = len(COL_NAMES)

    fig = plt.figure(figsize=(22, 2.6 * n_ch), facecolor='#0d1117')
    gs  = gridspec.GridSpec(n_ch, 2, hspace=0.35, wspace=0.08,
                            top=0.93, bottom=0.05, left=0.06, right=0.98)

    fig.suptitle(
        f'PHM 2010 — Outil {tool.upper()} — Comparaison Passe {num_a} vs Passe {num_b}\n'
        f'Fenetre : {WINDOW_MS} ms  |  fs = {FS:,} Hz',
        fontsize=12, color='white', fontweight='bold'
    )

    for col_idx, (sig, num, vb_label) in enumerate(
            [(sig_a, num_a, vb_a), (sig_b, num_b, vb_b)]):

        for row_idx, (ch, color) in enumerate(zip(COL_NAMES, COLORS)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            y  = sig[ch].values[:n_show]
            rms = np.sqrt(np.mean(y ** 2))
            pk  = y.max() - y.min()

            ax.plot(t_ms, y, color=color, linewidth=0.7, alpha=0.9)
            ax.fill_between(t_ms, y, alpha=0.08, color=color)
            style_ax(ax)

            # Titre de colonne (1re ligne seulement)
            if row_idx == 0:
                ax.set_title(
                    f'Passe {num} / {len(files)}{vb_label}',
                    color='white', fontsize=11, fontweight='bold', pad=8
                )

            # Label Y seulement à gauche
            if col_idx == 0:
                ax.set_ylabel(CHANNEL_LABELS.get(ch, ch), color='white', fontsize=8.5)
            else:
                ax.set_yticklabels([])

            ax.text(0.01, 0.91, f'RMS={rms:.3f}  Pk-Pk={pk:.3f}',
                    transform=ax.transAxes, fontsize=7, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#222', alpha=0.6))

            if row_idx < n_ch - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Temps (ms)', color='white', fontsize=8.5)

    out_name = f'compare_{tool}_pass{num_a:03d}_vs_pass{num_b:03d}.png'
    plt.savefig(out_name, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    print(f"[OK]   Comparaison sauvegardee -> {out_name}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Graphe time serie PHM 2010 — une ou deux passes comparees'
    )
    parser.add_argument('--dataset_path', type=str, default='./dataset_4',
                        help='Chemin vers le dossier dataset_4')
    parser.add_argument('--tool', type=str, default='c1',
                        help='Outil a visualiser (c1, c4, c6 ...)')
    parser.add_argument('--pass_idx', type=int, default=0,
                        help='Index de la passe principale (0 = passe 1)')
    parser.add_argument('--compare_idx', type=int, default=None,
                        help='Index de la 2e passe a comparer (ex: 149 pour passe 150). '
                             'Genere aussi une figure cote a cote.')
    parser.add_argument('--also_pass_150', action='store_true',
                        help='Generer egalement la passe 150 (index 149, 0-based).')
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"[ERROR] Dossier introuvable : {args.dataset_path}")
        return

    # Toujours générer le plot individuel de la passe principale
    plot_pass(args.dataset_path, args.tool, args.pass_idx)

    # Si l'option aussi pour la passe 150 est demandée et aucune passe de comparaison
    # explicite n'est fournie, on définit compare_idx = 149.
    if args.also_pass_150 and args.compare_idx is None:
        args.compare_idx = 149

    # Si une deuxième passe est demandée
    if args.compare_idx is not None:
        if args.compare_idx == args.pass_idx:
            print("[INFO] Les deux passes sont identiques, pas de comparaison.")
        else:
            # Plot individuel de la passe 2 (même style)
            plot_pass(args.dataset_path, args.tool, args.compare_idx)
            # Figure côte à côte
            plot_comparison(args.dataset_path, args.tool,
                            args.pass_idx, args.compare_idx)


if __name__ == '__main__':
    main()