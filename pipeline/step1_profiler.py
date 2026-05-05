"""
step1_profiler.py
─────────────────
Analyse automatiquement un dataset CNC et produit un profil complet.

Détecte :
  - Type de dataset (signaux bruts PHM / features tabulaires / autre)
  - Colonnes cible, features, métadonnées
  - Déséquilibre de classes
  - Leakage potentiel
  - Qualité des données (NaN, variance, doublons)

Sortie :
  - dataset_profile.json  → utilisé par toutes les étapes suivantes

Usage :
    python step1_profiler.py --dataset dataset_4/phm2010_features.csv
    python step1_profiler.py --dataset dataset_2/Exp2.csv
    python step1_profiler.py --dataset dataset_4/  (dossier de signaux bruts PHM)
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

# ─────────────────────────────────────────────────────────────────────────────
#  RÉSOLUTION DE LA RACINE DU PROJET
#  Si le script est dans un sous-dossier (ex: pipeline/), on remonte d'un
#  niveau pour que les chemins relatifs (dataset_4/, etc.) fonctionnent.
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FOLDER = os.path.basename(SCRIPT_DIR).lower()

SUBFOLDER_NAMES = ('pipeline', 'scripts', 'src', 'code', 'pipeline_cnc')
if SCRIPT_FOLDER in SUBFOLDER_NAMES:
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR


def resolve_path(path: str) -> str:
    """
    Résout un chemin en testant dans l'ordre :
      1. Tel quel (absolu ou relatif au CWD)
      2. Relatif à la racine du projet
      3. Relatif au dossier du script
    Retourne le premier qui existe, sinon le chemin tel quel.
    """
    if os.path.exists(path):
        return path
    candidate_root   = os.path.join(PROJECT_ROOT, path)
    candidate_script = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(candidate_root):
        return candidate_root
    if os.path.exists(candidate_script):
        return candidate_script
    return path  # sera géré par la vérif d'existence plus loin

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
WEAR_KEYWORDS = [
    'etat', 'tcond', 'tool_cond', 'tool_state', 'wear', 'vbb', 'vb_mean',
    'usure', 'flank', 'toolwear', 'condition', 'label', 'class', 'target',
    'output', 'state', 'status',
]

META_KEYWORDS = [
    'condition', 'outil', 'tool', 'id', 'index', 'passe', 'pass',
    'cut', 'exp', 'run', 'trial', 'sample', 'timestamp', 'time',
]

PHM_SIGNAL_COLS  = ['force_x', 'force_y', 'force_z', 'vib_x', 'vib_y', 'vib_z', 'ae']
LEAKAGE_THRESH   = 0.98
IMBALANCE_THRESH = 0.70   # si la classe majoritaire dépasse 70% → déséquilibre

# ─────────────────────────────────────────────────────────────────────────────
#  DÉTECTION DU TYPE DE DATASET
# ─────────────────────────────────────────────────────────────────────────────
def detect_dataset_type(path: str) -> dict:
    """
    Retourne un dict avec :
      - 'type'  : 'phm_raw' | 'tabular_features' | 'tabular_mixed'
      - 'files' : liste des fichiers CSV (si PHM brut)
      - 'main_csv' : chemin du CSV principal (si tabulaire)
    """
    # ── Dossier → PHM signaux bruts ─────────────────────────────────────
    if os.path.isdir(path):
        signal_files = sorted(glob.glob(
            os.path.join(path, '**', '*.csv'), recursive=True
        ))
        signal_files = [f for f in signal_files
                        if 'wear' not in os.path.basename(f).lower()
                        and 'features' not in os.path.basename(f).lower()]

        if signal_files:
            # Vérifie si les fichiers ressemblent à des signaux PHM
            sample = pd.read_csv(signal_files[0], header=None, nrows=5)
            if sample.shape[1] in (6, 7):
                return {'type': 'phm_raw', 'files': signal_files, 'main_csv': None}

        # Cherche un CSV tabulaire dans le dossier
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        if csv_files:
            path = csv_files[0]  # prend le premier

    # ── Fichier CSV ───────────────────────────────────────────────────────
    if not os.path.isfile(path):
        return {'type': 'unknown', 'files': [], 'main_csv': None}

    df_head = pd.read_csv(path, nrows=5)

    # PHM features déjà extraites (nombreuses colonnes force_*/vib_*)
    feature_cols = [c for c in df_head.columns
                    if any(c.startswith(p) for p in
                           ['force_', 'vib_', 'ae_', 'force_x', 'force_y'])]
    if len(feature_cols) > 10:
        return {'type': 'tabular_features', 'files': [path], 'main_csv': path}

    # CSV CNC Turning (colonnes mixtes process + signaux)
    return {'type': 'tabular_mixed', 'files': [path], 'main_csv': path}


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSE DU CSV TABULAIRE
# ─────────────────────────────────────────────────────────────────────────────
def detect_target(df: pd.DataFrame) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in WEAR_KEYWORDS:
        if kw in cols_lower:
            return cols_lower[kw]
    # Fallback : dernière colonne non-numérique ou dernière numérique
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        return obj_cols[-1]
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    return num_cols[-1] if num_cols else None


def detect_task(series: pd.Series) -> str:
    if series.dtype == object:
        return 'classification'
    n_u   = series.nunique()
    ratio = n_u / max(len(series), 1)
    return 'classification' if (n_u <= 15 or ratio < 0.05) else 'regression'


def detect_meta_columns(df: pd.DataFrame, target_col: str) -> list:
    """Colonnes qui ne sont pas des features physiques (ID, numéro, texte...)."""
    meta = []
    for col in df.columns:
        if col == target_col:
            continue
        col_l = col.lower()
        # Nom explicitement méta
        if any(kw in col_l for kw in META_KEYWORDS):
            meta.append(col)
            continue
        # Colonnes object (string) non target
        if df[col].dtype == object:
            meta.append(col)
            continue
        # Colonnes entières avec peu de valeurs uniques potentiellement IDs
        if df[col].dtype in (np.int32, np.int64) and df[col].nunique() < 20:
            meta.append(col)
    return meta


def compute_correlations(df: pd.DataFrame, features: list,
                          target_col: str, task: str) -> dict:
    """Corrélation |r| de chaque feature avec la target."""
    from sklearn.preprocessing import LabelEncoder
    if task == 'classification' or df[target_col].dtype == object:
        le  = LabelEncoder()
        y   = pd.Series(le.fit_transform(df[target_col].fillna('NA').astype(str)),
                        index=df.index, name='__target__')
    else:
        y = df[target_col].rename('__target__')

    corr_map = {}
    for col in features:
        try:
            valid = df[[col]].join(y).dropna()
            r     = abs(valid[col].corr(valid['__target__']))
            corr_map[col] = round(float(r), 5) if not np.isnan(r) else None
        except Exception:
            corr_map[col] = None
    return corr_map


def analyze_class_balance(series: pd.Series) -> dict:
    vc    = series.value_counts()
    total = len(series)
    dist  = {str(k): int(v) for k, v in vc.items()}
    pct   = {str(k): round(v / total, 4) for k, v in vc.items()}
    majority_pct = vc.iloc[0] / total

    return {
        'n_classes'     : int(series.nunique()),
        'classes'       : list(dist.keys()),
        'counts'        : dist,
        'percentages'   : pct,
        'majority_pct'  : round(float(majority_pct), 4),
        'is_imbalanced' : bool(majority_pct > IMBALANCE_THRESH),
        'imbalance_ratio': round(float(vc.iloc[0] / max(vc.iloc[-1], 1)), 2),
    }


def analyze_data_quality(df: pd.DataFrame, features: list) -> dict:
    total_cells = len(df) * len(features)
    nan_counts  = {col: int(df[col].isna().sum()) for col in features
                   if df[col].isna().sum() > 0}
    zero_var    = [col for col in features if df[col].std() < 1e-9]
    duplicates  = int(df.duplicated().sum())

    return {
        'n_rows'        : int(len(df)),
        'n_features'    : len(features),
        'total_nan'     : int(df[features].isna().sum().sum()),
        'nan_pct'       : round(df[features].isna().sum().sum() / max(total_cells, 1) * 100, 3),
        'nan_by_column' : nan_counts,
        'zero_variance' : zero_var,
        'n_duplicates'  : duplicates,
        'rows_after_dropna': int(df[features].dropna().shape[0]),
    }


def profile_tabular(path: str, dataset_type: str) -> dict:
    """Profil complet d'un CSV tabulaire."""
    print(f"  Lecture : {path}")
    df = pd.read_csv(path)
    print(f"  {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # ── Target ────────────────────────────────────────────────────────────
    target_col = detect_target(df)
    task       = detect_task(df[target_col]) if target_col else 'unknown'

    # ── Colonnes méta ─────────────────────────────────────────────────────
    meta_cols  = detect_meta_columns(df, target_col) if target_col else []

    # ── Features ──────────────────────────────────────────────────────────
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    features  = [c for c in num_cols
                 if c != target_col and c not in meta_cols]
    zero_var  = [c for c in features if df[c].std() < 1e-9]
    features  = [c for c in features if c not in zero_var]

    # ── Corrélations ──────────────────────────────────────────────────────
    print(f"  Calcul des corrélations ({len(features)} features)...")
    corr_map = compute_correlations(df, features, target_col, task) if target_col else {}
    leaky    = [c for c, r in corr_map.items()
                if r is not None and r >= LEAKAGE_THRESH]

    # ── Qualité ───────────────────────────────────────────────────────────
    quality  = analyze_data_quality(df, features)

    # ── Balance ───────────────────────────────────────────────────────────
    balance  = None
    if task == 'classification' and target_col:
        balance = analyze_class_balance(df[target_col].dropna())

    # ── Stats descriptives ─────────────────────────────────────────────────
    desc = {}
    for col in features[:20]:   # top 20 pour ne pas exploser le JSON
        s = df[col].describe()
        desc[col] = {k: round(float(v), 5) for k, v in s.items()}

    return {
        'dataset_type'   : dataset_type,
        'path'           : os.path.abspath(path),
        'filename'       : os.path.basename(path),
        'target_column'  : target_col,
        'task'           : task,
        'meta_columns'   : meta_cols,
        'features'       : features,
        'features_leaky' : leaky,
        'features_recommended': [c for c in features if c not in leaky],
        'correlations'   : corr_map,
        'data_quality'   : quality,
        'class_balance'  : balance,
        'descriptive_stats': desc,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSE PHM SIGNAUX BRUTS
# ─────────────────────────────────────────────────────────────────────────────
def profile_phm_raw(dataset_path: str, files: list) -> dict:
    """Profil d'un dataset de signaux bruts PHM 2010."""
    print(f"  {len(files)} fichiers signal détectés")

    # Détecte les outils (sous-dossiers)
    tools = sorted(set(
        os.path.basename(os.path.dirname(f)) for f in files
    ))

    # Compte les passes par outil
    passes_per_tool = {}
    for t in tools:
        tool_files = [f for f in files
                      if os.path.basename(os.path.dirname(f)) == t]
        passes_per_tool[t] = len(tool_files)

    # Lit un fichier sample pour les stats
    sample_file = files[0]
    df_sample   = pd.read_csv(sample_file, header=None,
                               names=PHM_SIGNAL_COLS[:7])
    n_samples_per_pass = len(df_sample)
    fs                 = 50_000  # Hz PHM 2010

    # Cherche les fichiers wear
    wear_files = glob.glob(
        os.path.join(dataset_path, '**', '*wear*.csv'), recursive=True
    )
    wear_info = {}
    for wf in wear_files:
        tool_name = os.path.splitext(os.path.basename(wf))[0].replace('_wear', '')
        df_w = pd.read_csv(wf, header=None)
        df_w = df_w.apply(pd.to_numeric, errors='coerce').dropna(how='all')
        if df_w.shape[1] >= 2:
            wear_info[tool_name] = {
                'n_measurements': int(len(df_w)),
                'vb_min'  : round(float(df_w.iloc[:, 1:].min().min()), 4),
                'vb_max'  : round(float(df_w.iloc[:, 1:].max().max()), 4),
                'vb_mean' : round(float(df_w.iloc[:, 1:].mean().mean()), 4),
            }

    # Vérifie si les features sont déjà extraites
    feature_csv = glob.glob(
        os.path.join(dataset_path, '*features*.csv')
    )

    return {
        'dataset_type'        : 'phm_raw',
        'path'                : os.path.abspath(dataset_path),
        'filename'            : os.path.basename(dataset_path),
        'n_signal_files'      : len(files),
        'tools'               : tools,
        'passes_per_tool'     : passes_per_tool,
        'signal_channels'     : PHM_SIGNAL_COLS,
        'n_channels'          : len(PHM_SIGNAL_COLS),
        'n_samples_per_pass'  : n_samples_per_pass,
        'sampling_rate_hz'    : fs,
        'duration_per_pass_ms': round(n_samples_per_pass / fs * 1000, 2),
        'wear_files'          : [os.path.basename(f) for f in wear_files],
        'wear_info'           : wear_info,
        'features_already_extracted': len(feature_csv) > 0,
        'feature_csv'         : [os.path.basename(f) for f in feature_csv],
        'next_step_required'  : 'feature_extraction'
            if not feature_csv else 'use_existing_features',
        # PHM raw ne peut pas aller en benchmark directement
        'target_column'       : 'etat',
        'task'                : 'classification',
        'features'            : None,
        'class_balance'       : None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RÉSUMÉ CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(profile: dict) -> None:
    W = 62
    print(f'\n{"═" * W}')
    print(f'  PROFIL DU DATASET')
    print(f'{"═" * W}')
    print(f'  Fichier      : {profile.get("filename")}')
    print(f'  Type         : {profile.get("dataset_type")}')
    print(f'  Tâche        : {profile.get("task", "N/A").upper()}')
    print(f'  Cible        : {profile.get("target_column", "N/A")}')

    if profile.get('dataset_type') == 'phm_raw':
        print(f'  Outils       : {profile["tools"]}')
        print(f'  Total passes : {profile["n_signal_files"]}')
        print(f'  Channels     : {profile["signal_channels"]}')
        print(f'  Features CSV : {profile["feature_csv"] or "aucun — extraction nécessaire"}')
        print(f'\n  ➡  Prochaine étape : step2_feature_engine.py')

    else:
        q = profile.get('data_quality', {})
        print(f'  Lignes       : {q.get("n_rows", "N/A")}')
        print(f'  Features     : {q.get("n_features", "N/A")}')
        print(f'  NaN          : {q.get("total_nan", 0)} ({q.get("nan_pct", 0):.2f}%)')
        print(f'  Doublons     : {q.get("n_duplicates", 0)}')
        print(f'  Colonnes méta: {profile.get("meta_columns", [])}')

        leaky = profile.get('features_leaky', [])
        if leaky:
            print(f'\n  ⚠  LEAKAGE ({len(leaky)} colonnes) :')
            for c in leaky:
                r = profile['correlations'].get(c)
                print(f'     • {c}  (|r| = {r})')

        bal = profile.get('class_balance')
        if bal:
            print(f'\n  Distribution de "{profile["target_column"]}" :')
            for cls, pct in bal['percentages'].items():
                bar = '█' * int(pct * 30)
                print(f'     {cls:<20} {bar} {pct*100:.1f}%  ({bal["counts"][cls]} ex.)')
            if bal['is_imbalanced']:
                print(f'\n  ⚠  DÉSÉQUILIBRE DÉTECTÉ')
                print(f'     Ratio majority/minority = {bal["imbalance_ratio"]}x')
                print(f'     → step3_balancer.py sera nécessaire')
            else:
                print(f'  ✅ Classes équilibrées')

        rec = profile.get('features_recommended', [])
        print(f'\n  Features recommandées : {len(rec)} / {q.get("n_features", "?")}')
        print(f'\n  ➡  Prochaine étape : step2_feature_engine.py')

    print(f'{"═" * W}\n')


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Step 1 — Profiler automatique de dataset CNC'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Chemin vers le CSV ou le dossier dataset')
    parser.add_argument('--output',  type=str, default=None,
                        help='Dossier de sortie du JSON (défaut: même dossier que le script)')
    args = parser.parse_args()

    # Résolution des chemins (fonctionne depuis pipeline/ ou depuis la racine)
    args.dataset = resolve_path(args.dataset)
    if args.output is None:
        args.output = SCRIPT_DIR   # JSON à côté du script dans pipeline/
    else:
        args.output = resolve_path(args.output)

    if not os.path.exists(args.dataset):
        print(f'[ERROR] Introuvable : {args.dataset}')
        print(f'        Racine projet testée : {PROJECT_ROOT}')
        sys.exit(1)

    print(f'\n{"═" * 62}')
    print(f'  STEP 1 — PROFILER DE DATASET CNC')
    print(f'{"═" * 62}\n')
    print(f'  Racine projet : {PROJECT_ROOT}')
    print(f'  Analyse de    : {args.dataset}\n')

    # ── Détection du type ────────────────────────────────────────────────
    dt = detect_dataset_type(args.dataset)
    print(f'  Type détecté : {dt["type"]}')

    # ── Profil selon le type ─────────────────────────────────────────────
    if dt['type'] == 'phm_raw':
        profile = profile_phm_raw(args.dataset, dt['files'])
    else:
        csv_path = dt['main_csv'] or args.dataset
        profile  = profile_tabular(csv_path, dt['type'])

    # ── Métadonnées communes ─────────────────────────────────────────────
    profile['generated_at'] = datetime.now().isoformat()
    profile['pipeline_step'] = 1

    # ── Sauvegarde JSON ──────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, 'dataset_profile.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, default=str, ensure_ascii=False)

    print_summary(profile)
    print(f'  💾 Profil sauvegardé → {os.path.abspath(out_path)}')
    print(f'  ✅ Step 1 terminé\n')


if __name__ == '__main__':
    main()
