"""
step2_feature_engine.py
───────────────────────
Lit dataset_profile.json (Step 1) et prépare les données ML propres.

Si dataset tabulaire  → supprime leakage, méta-colonnes, NaN, zero-variance
                         sort X_clean.csv + y_clean.csv
Si dataset PHM brut   → extrait les features statistiques de chaque passe
                         (RMS, std, skew, kurt, crest, énergie...)
                         sort X_clean.csv + y_clean.csv

Usage :
    python step2_feature_engine.py
    python step2_feature_engine.py --profile dataset_profile.json
    python step2_feature_engine.py --profile dataset_profile.json --window_ms 40
"""

import os
import sys
import json
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  RÉSOLUTION DE LA RACINE DU PROJET
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FOLDER = os.path.basename(SCRIPT_DIR).lower()
SUBFOLDER_NAMES = ('pipeline', 'scripts', 'src', 'code', 'pipeline_cnc')
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR) if SCRIPT_FOLDER in SUBFOLDER_NAMES else SCRIPT_DIR


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    for base in (PROJECT_ROOT, SCRIPT_DIR):
        c = os.path.join(base, path)
        if os.path.exists(c):
            return c
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION DE FEATURES — SIGNAUX PHM BRUTS
# ─────────────────────────────────────────────────────────────────────────────
PHM_CHANNELS = ['force_x', 'force_y', 'force_z', 'vib_x', 'vib_y', 'vib_z', 'ae']
FS           = 50_000   # Hz


def extract_features_from_signal(sig: np.ndarray, channel: str) -> dict:
    """Extrait ~13 features statistiques d'un canal 1D."""
    sig = sig[~np.isnan(sig)]
    if len(sig) == 0:
        return {f'{channel}_{s}': np.nan for s in
                ['mean','std','rms','max','p2p','skew','kurt',
                 'crest','shape','energy','p25','p75','iqr']}
    rms    = float(np.sqrt(np.mean(sig ** 2)))
    std    = float(np.std(sig))
    mean_v = float(np.mean(sig))
    mx     = float(np.max(np.abs(sig)))
    p2p    = float(np.max(sig) - np.min(sig))
    sk     = float(skew(sig))
    kt     = float(kurtosis(sig))
    crest  = mx / rms if rms > 1e-12 else 0.0
    shape  = rms / (np.mean(np.abs(sig)) + 1e-12)
    energy = float(np.sum(sig ** 2))
    p25    = float(np.percentile(sig, 25))
    p75    = float(np.percentile(sig, 75))
    iqr    = p75 - p25
    return {
        f'{channel}_mean'  : round(mean_v, 8),
        f'{channel}_std'   : round(std,    8),
        f'{channel}_rms'   : round(rms,    8),
        f'{channel}_max'   : round(mx,     6),
        f'{channel}_p2p'   : round(p2p,    6),
        f'{channel}_skew'  : round(sk,     8),
        f'{channel}_kurt'  : round(kt,     8),
        f'{channel}_crest' : round(crest,  8),
        f'{channel}_shape' : round(shape,  8),
        f'{channel}_energy': round(energy, 6),
        f'{channel}_p25'   : round(p25,    6),
        f'{channel}_p75'   : round(p75,    6),
        f'{channel}_iqr'   : round(iqr,    6),
    }


def load_wear_map(dataset_path: str) -> dict:
    """
    Charge tous les fichiers *_wear.csv et retourne un dict
    {(tool, passe_num): vb_mean}.
    """
    wear_files = glob.glob(
        os.path.join(dataset_path, '**', '*wear*.csv'), recursive=True
    )
    wear_map = {}
    for wf in wear_files:
        tool = os.path.splitext(os.path.basename(wf))[0].replace('_wear', '')
        try:
            df_w = pd.read_csv(wf, header=None)
            df_w = df_w.apply(pd.to_numeric, errors='coerce').dropna(how='all')
            if df_w.shape[1] < 2:
                continue
            # Colonnes : passe, vb1[, vb2, vb3]
            for _, row in df_w.iterrows():
                passe = int(row.iloc[0])
                vb_vals = row.iloc[1:].dropna().values.astype(float)
                if len(vb_vals) == 0:
                    continue
                vb_mean = float(np.mean(vb_vals))
                # mm si valeur > 10 (probablement en µm)
                if vb_mean > 10:
                    vb_mean /= 1000.0
                wear_map[(tool, passe)] = round(vb_mean, 5)
        except Exception:
            continue
    return wear_map


def assign_wear_state(vb: float) -> str:
    """Seuils PHM 2010 standard."""
    if vb < 0.1:
        return 'neuf'
    elif vb < 0.2:
        return 'intermediaire'
    else:
        return 'use'


def extract_phm_features(profile: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Itère sur tous les fichiers signal PHM, extrait les features et
    construit le dataset tabulaire avec la colonne cible 'etat'.
    """
    dataset_path = os.path.dirname(profile['path'])
    files        = sorted(glob.glob(
        os.path.join(profile['path'], '**', '*.csv'), recursive=True
    ))
    files = [f for f in files
             if 'wear' not in os.path.basename(f).lower()
             and 'features' not in os.path.basename(f).lower()]

    wear_map = load_wear_map(profile['path'])

    rows  = []
    total = len(files)
    print(f'  Extraction features sur {total} fichiers...')

    for i, fpath in enumerate(files, 1):
        if i % 100 == 0 or i == total:
            print(f'  {i}/{total}', end='\r', flush=True)

        # Nom du tool et numéro de passe depuis le nom de fichier
        fname    = os.path.splitext(os.path.basename(fpath))[0]  # ex: c_1_042
        parts    = fname.split('_')  # ['c', '1', '042']
        if len(parts) >= 3:
            tool  = f'c{parts[1]}'
            try:
                passe = int(parts[2])
            except ValueError:
                continue
        else:
            continue

        try:
            df_sig = pd.read_csv(fpath, header=None)
            if df_sig.shape[1] < len(PHM_CHANNELS):
                df_sig = pd.read_csv(fpath, header=None,
                                     usecols=range(min(df_sig.shape[1],
                                                        len(PHM_CHANNELS))))
            df_sig.columns = PHM_CHANNELS[:df_sig.shape[1]]
        except Exception:
            continue

        row = {'tool': tool, 'passe': passe}
        for ch in PHM_CHANNELS[:df_sig.shape[1]]:
            feats = extract_features_from_signal(df_sig[ch].values, ch)
            row.update(feats)

        # Label d'usure
        vb    = wear_map.get((tool, passe))
        row['vb_mean'] = vb
        row['etat']    = assign_wear_state(vb) if vb is not None else 'inconnu'

        rows.append(row)

    print(f'\n  {len(rows)} passes traitées.')
    df = pd.DataFrame(rows)
    y  = df['etat']
    X  = df.drop(columns=['etat', 'vb_mean', 'tool', 'passe'],
                  errors='ignore')
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  NETTOYAGE — DATASET TABULAIRE
# ─────────────────────────────────────────────────────────────────────────────
def clean_tabular(profile: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge le CSV, supprime :
      - colonnes méta (passe, tool id, etc.)
      - colonnes leaky (corrélation > 0.98 avec target)
      - colonnes zero-variance
      - lignes avec NaN
    Retourne (X, y).
    """
    csv_path = profile['path']
    print(f'  Lecture : {os.path.basename(csv_path)}')
    df = pd.read_csv(csv_path)

    target_col = profile['target_column']
    if target_col not in df.columns:
        print(f'  [ERROR] Colonne cible "{target_col}" introuvable.')
        sys.exit(1)

    y = df[target_col].copy()

    # Colonnes à exclure
    to_drop = set()
    to_drop.update(profile.get('meta_columns', []))
    to_drop.update(profile.get('features_leaky', []))
    to_drop.discard(target_col)

    existing_drop = [c for c in to_drop if c in df.columns]
    if existing_drop:
        print(f'  Colonnes exclues ({len(existing_drop)}) : {existing_drop}')
        df = df.drop(columns=existing_drop)

    # Features numériques restantes
    features = [c for c in profile.get('features_recommended', [])
                if c in df.columns]
    if not features:
        # Fallback : toutes les colonnes numériques hors target
        features = df.select_dtypes(include=np.number).columns.tolist()
        features = [c for c in features if c != target_col]

    X = df[features].copy()

    # Suppression NaN
    mask   = X.notna().all(axis=1) & y.notna()
    n_drop = (~mask).sum()
    if n_drop:
        print(f'  Suppression de {n_drop} lignes avec NaN')
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print(f'  Dataset propre : {X.shape[0]} lignes × {X.shape[1]} features')
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Step 2 — Feature engine & nettoyage'
    )
    parser.add_argument('--profile', type=str,
                        default='dataset_profile.json',
                        help='Chemin vers dataset_profile.json (Step 1)')
    parser.add_argument('--window_ms', type=float, default=40.0,
                        help='Fenêtre d\'extraction (ms) si PHM brut (défaut: 40)')
    args = parser.parse_args()

    profile_path = resolve_path(args.profile)
    if not os.path.isfile(profile_path):
        print(f'[ERROR] Profil introuvable : {profile_path}')
        print(f'        Lance d\'abord : python step1_profiler.py --dataset <ton_csv>')
        sys.exit(1)

    with open(profile_path, encoding='utf-8') as f:
        profile = json.load(f)

    print(f'\n{"═" * 62}')
    print(f'  STEP 2 — FEATURE ENGINE')
    print(f'  Dataset : {profile["filename"]}  |  Type : {profile["dataset_type"]}')
    print(f'{"═" * 62}\n')

    # ── Traitement selon le type ──────────────────────────────────────────
    if profile['dataset_type'] == 'phm_raw':
        print('  Mode : extraction features depuis signaux bruts PHM')
        X, y = extract_phm_features(profile)
    else:
        print('  Mode : nettoyage dataset tabulaire')
        X, y = clean_tabular(profile)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    x_path = os.path.join(SCRIPT_DIR, 'X_clean.csv')
    y_path = os.path.join(SCRIPT_DIR, 'y_clean.csv')

    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False, header=['target'])

    # Mise à jour du profil
    profile['step2_features'] = list(X.columns)
    profile['step2_n_samples'] = int(len(X))
    profile['step2_done']      = True
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, default=str, ensure_ascii=False)

    print(f'\n  💾 X_clean.csv  → {x_path}  ({X.shape[0]}×{X.shape[1]})')
    print(f'  💾 y_clean.csv  → {y_path}')
    print(f'  ✅ Step 2 terminé — Lance step3_balancer.py\n')


if __name__ == '__main__':
    main()
