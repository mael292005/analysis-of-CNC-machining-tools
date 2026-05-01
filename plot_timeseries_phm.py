"""
plot_timeseries_phm.py
─────────────────────
Visualise une passe brute du dataset PHM 2010 (signaux temporels).
Usage :
    python plot_timeseries_phm.py --dataset_path ./dataset_4
    python plot_timeseries_phm.py --dataset_path ./dataset_4 --tool c1 --pass_idx 0
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Configuration ───────────────────────────────────────────────────────────
FS          = 50_000                          # Fréquence d'échantillonnage (Hz)
COL_NAMES   = ['force_x', 'force_y', 'force_z', 'vib_x', 'vib_y', 'vib_z', 'ae']
WINDOW_MS   = 40                              # Fenêtre d'affichage en ms
COLORS      = ['#1565C0', '#E65100', '#2E7D32', '#6A1B9A', '#C62828', '#00838F', '#F57F17']

CHANNEL_LABELS = {
    'force_x': 'Force X (N)',
    'force_y': 'Force Y (N)',
    'force_z': 'Force Z (N)',
    'vib_x'  : 'Vibration X (m/s²)',
    'vib_y'  : 'Vibration Y (m/s²)',
    'vib_z'  : 'Vibration Z (m/s²)',
    'ae'     : 'Acoustic Emission (V)',
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def find_signal_files(dataset_path: str, tool: str) -> list[str]:
    """Cherche tous les fichiers CSV de signal (hors wear) pour un outil."""
    patterns = [
        os.path.join(dataset_path, tool, '**', '*.csv'),
        os.path.join(dataset_path, '**', f'{tool}_*.csv'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if 'wear' not in os.path.basename(f).lower()]
    return sorted(set(files))


def load_signal(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, header=None, names=COL_NAMES)


def load_wear(dataset_path: str, tool: str) -> pd.DataFrame | None:
    """Charge le fichier d'usure si disponible."""
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


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_pass(dataset_path: str, tool: str = 'c1', pass_idx: int = 0) -> None:
    """Affiche tous les canaux d'une passe choisie."""
    files = find_signal_files(dataset_path, tool)
    if not files:
        print(f"[ERROR] Aucun fichier signal trouvé pour l'outil '{tool}' dans '{dataset_path}'")
        print("        Outils disponibles :", end=' ')
        subdirs = [d for d in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, d))]
        print(subdirs if subdirs else "aucun sous-dossier trouvé")
        return

    pass_idx = max(0, min(pass_idx, len(files) - 1))
    filepath  = files[pass_idx]
    passe_num = pass_idx + 1

    print(f"[INFO] Outil : {tool.upper()} | Fichier passe {passe_num}/{len(files)}")
    print(f"       Fichier : {os.path.basename(filepath)}")

    sig    = load_signal(filepath)
    n_pts  = len(sig)
    dur_ms = n_pts / FS * 1000
    n_show = int(WINDOW_MS * FS / 1000)
    t_ms   = np.arange(n_show) / FS * 1000

    # ─ Wear label (si disponible) ──
    wear_df  = load_wear(dataset_path, tool)
    vb_label = ''
    if wear_df is not None and passe_num in wear_df['passe'].values:
        vb = wear_df.loc[wear_df['passe'] == passe_num, 'vb_mean'].values[0]
        vb_label = f'  |  VB = {vb:.3f} mm'

    # ─ Layout ─────────────────────────────────────────────────────────────
    n_ch = len(COL_NAMES)
    fig  = plt.figure(figsize=(14, 2.8 * n_ch), facecolor='#0d1117')
    gs   = gridspec.GridSpec(n_ch, 1, hspace=0.35)

    suptitle = (
        f'PHM 2010 — Outil {tool.upper()} — Passe {passe_num}/{len(files)}{vb_label}\n'
        f'Fenêtre : {WINDOW_MS} ms   |   Durée totale : {dur_ms:.0f} ms   |   '
        f'fs = {FS:,} Hz   |   {n_pts:,} points/canal'
    )
    fig.suptitle(suptitle, fontsize=11, color='white', fontweight='bold', y=1.01)

    for i, (col, color) in enumerate(zip(COL_NAMES, COLORS)):
        ax = fig.add_subplot(gs[i])
        y  = sig[col].values[:n_show]

        ax.plot(t_ms, y, color=color, linewidth=0.7, alpha=0.9)
        ax.fill_between(t_ms, y, alpha=0.08, color=color)

        # Stats summary
        rms   = np.sqrt(np.mean(y ** 2))
        pk_pk = y.max() - y.min()
        ax.set_ylabel(CHANNEL_LABELS.get(col, col), color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.set_facecolor('#0d1117')
        ax.grid(True, alpha=0.15, color='white', linewidth=0.5)

        # Annotation RMS
        ax.text(0.01, 0.92, f'RMS={rms:.3f}  Pk-Pk={pk_pk:.3f}',
                transform=ax.transAxes, fontsize=7.5, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#222', alpha=0.6))

        if i < n_ch - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Temps (ms)', color='white', fontsize=9)

    plt.tight_layout()
    out_name = f'timeseries_{tool}_pass{passe_num:03d}.png'
    plt.savefig(out_name, dpi=140, bbox_inches='tight', facecolor='#0d1117')
    print(f"[OK]   Graphe sauvegardé → {out_name}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Graphe time série PHM 2010 — une passe, tous les canaux'
    )
    parser.add_argument('--dataset_path', type=str, default='./dataset_4',
                        help='Chemin vers le dossier dataset_4')
    parser.add_argument('--tool', type=str, default='c1',
                        help='Outil à visualiser (c1, c4, c6 …)')
    parser.add_argument('--pass_idx', type=int, default=0,
                        help='Index de la passe (0 = première passe)')
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"[ERROR] Dossier introuvable : {args.dataset_path}")
        return

    plot_pass(args.dataset_path, args.tool, args.pass_idx)


if __name__ == '__main__':
    main()
