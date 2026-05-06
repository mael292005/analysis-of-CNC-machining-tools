"""
step3_balancer.py
─────────────────
Lit X_clean.csv + y_clean.csv (Step 2) et gère le déséquilibre de classes.

Stratégies disponibles :
  - auto      : choisit automatiquement selon le ratio de déséquilibre
  - smote     : sur-échantillonnage synthétique (SMOTE)
  - weights   : calcule class_weight pour les modèles (pas de rééchantillonnage)
  - none      : pas de correction (déconseillé si déséquilibre fort)

Sortie :
  - X_balanced.csv + y_balanced.csv  (si rééchantillonnage)
  - balancer_info.json               (class_weights + stratégie utilisée)
  - Met à jour dataset_profile.json

Usage :
    python step3_balancer.py
    python step3_balancer.py --strategy smote
    python step3_balancer.py --strategy weights
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

warnings.filterwarnings('ignore')

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
#  ANALYSE DU DÉSÉQUILIBRE
# ─────────────────────────────────────────────────────────────────────────────
def analyze_balance(y: pd.Series) -> dict:
    counts  = Counter(y)
    total   = len(y)
    maj     = max(counts.values())
    mn      = min(counts.values())
    ratio   = maj / mn if mn > 0 else float('inf')

    print(f'\n  Distribution actuelle :')
    for cls in sorted(counts):
        pct = counts[cls] / total * 100
        bar = '█' * int(pct / 2)
        print(f'    {str(cls):<20} {bar:<50} {pct:.1f}%  ({counts[cls]} ex.)')
    print(f'\n  Ratio majority/minority : {ratio:.1f}x')
    return {'counts': dict(counts), 'ratio': ratio, 'total': total}


def recommend_strategy(ratio: float, total: int) -> str:
    if ratio < 1.5:
        return 'none'
    elif ratio < 5.0:
        return 'weights'
    elif total < 500:
        return 'weights'   # SMOTE peu fiable sur petits datasets
    else:
        return 'smote'


# ─────────────────────────────────────────────────────────────────────────────
#  STRATÉGIE : CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(y: pd.Series) -> dict:
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    cw = {str(cls): round(float(w), 4) for cls, w in zip(classes, weights)}
    print(f'\n  Class weights calculés :')
    for cls, w in cw.items():
        print(f'    {cls:<20} → weight = {w}')
    return cw


# ─────────────────────────────────────────────────────────────────────────────
#  STRATÉGIE : SMOTE
# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print('\n  [INFO] imbalanced-learn non installé.')
        print('         Installation : pip install imbalanced-learn')
        print('         → Fallback sur la stratégie "weights"')
        return X, y

    counts = Counter(y)
    min_count = min(counts.values())

    # SMOTE nécessite au moins k_neighbors+1 exemples par classe
    k = min(5, min_count - 1)
    if k < 1:
        print(f'  [WARN] Classe minoritaire trop petite ({min_count} ex.) pour SMOTE.')
        print('         → Fallback sur "weights"')
        return X, y

    print(f'\n  Application SMOTE (k_neighbors={k})...')
    sm      = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = sm.fit_resample(X.values, y.values)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(y_res, name=y.name)

    print(f'  Avant SMOTE : {len(X)} exemples')
    print(f'  Après SMOTE : {len(X_res)} exemples')
    new_counts = Counter(y_res)
    for cls in sorted(new_counts):
        print(f'    {str(cls):<20} → {new_counts[cls]} ex.')

    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Step 3 — Gestion du déséquilibre de classes'
    )
    parser.add_argument('--strategy', type=str, default='auto',
                        choices=['auto', 'smote', 'weights', 'none'],
                        help='Stratégie de rééquilibrage (défaut: auto)')
    parser.add_argument('--profile', type=str,
                        default='dataset_profile.json')
    args = parser.parse_args()

    profile_path = resolve_path(args.profile)
    x_path       = resolve_path('X_clean.csv')
    y_path       = resolve_path('y_clean.csv')

    for p, name in [(profile_path, 'dataset_profile.json'),
                    (x_path, 'X_clean.csv'),
                    (y_path, 'y_clean.csv')]:
        if not os.path.isfile(p):
            print(f'[ERROR] Fichier manquant : {name}')
            print('        Lance d\'abord step2_feature_engine.py')
            sys.exit(1)

    with open(profile_path, encoding='utf-8') as f:
        profile = json.load(f)

    print(f'\n{"═" * 62}')
    print(f'  STEP 3 — ÉQUILIBRAGE DES CLASSES')
    print(f'  Dataset : {profile["filename"]}  |  Tâche : {profile["task"].upper()}')
    print(f'{"═" * 62}')

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)['target']

    # ── Régression : pas d'équilibrage nécessaire ─────────────────────────
    if profile.get('task') == 'regression':
        print('\n  Tâche de régression → pas d\'équilibrage nécessaire.')
        info = {'strategy': 'none', 'reason': 'regression task',
                'class_weights': None, 'resampled': False}
        _save_and_exit(X, y, info, profile, profile_path)
        return

    # ── Analyse ───────────────────────────────────────────────────────────
    bal = analyze_balance(y)

    # ── Choix de la stratégie ─────────────────────────────────────────────
    strategy = args.strategy
    if strategy == 'auto':
        strategy = recommend_strategy(bal['ratio'], bal['total'])
        print(f'\n  Stratégie recommandée : {strategy.upper()}')
        print(f'  (ratio={bal["ratio"]:.1f}x, n={bal["total"]})')
    else:
        print(f'\n  Stratégie choisie : {strategy.upper()}')

    # ── Application ───────────────────────────────────────────────────────
    class_weights = None
    resampled     = False

    if strategy == 'smote':
        X_out, y_out = apply_smote(X, y)
        resampled    = len(X_out) != len(X)
        # Calcule aussi les poids pour les modèles qui les acceptent
        class_weights = compute_class_weights(y)

    elif strategy == 'weights':
        X_out, y_out  = X, y
        class_weights = compute_class_weights(y)
        print('\n  Aucun rééchantillonnage — les poids seront passés aux modèles.')

    else:  # none
        X_out, y_out = X, y
        print('\n  Aucune correction appliquée.')

    # ── Sauvegarde ────────────────────────────────────────────────────────
    info = {
        'strategy'      : strategy,
        'class_weights' : class_weights,
        'resampled'     : resampled,
        'n_before'      : int(bal['total']),
        'n_after'       : int(len(X_out)),
        'imbalance_ratio': round(bal['ratio'], 2),
        'generated_at'  : datetime.now().isoformat(),
    }
    _save_and_exit(X_out, y_out, info, profile, profile_path)


def _save_and_exit(X, y, info, profile, profile_path):
    x_out = os.path.join(SCRIPT_DIR, 'X_balanced.csv')
    y_out = os.path.join(SCRIPT_DIR, 'y_balanced.csv')
    bi_out = os.path.join(SCRIPT_DIR, 'balancer_info.json')

    X.to_csv(x_out, index=False)
    y.to_csv(y_out, index=False, header=['target'])

    with open(bi_out, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, default=str)

    # Mise à jour du profil
    profile['step3_strategy']     = info['strategy']
    profile['step3_class_weights'] = info['class_weights']
    profile['step3_n_after']      = int(len(X))
    profile['step3_done']         = True
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, default=str, ensure_ascii=False)

    print(f'\n  💾 X_balanced.csv   → {x_out}')
    print(f'  💾 y_balanced.csv   → {y_out}')
    print(f'  💾 balancer_info.json → {bi_out}')
    print(f'  ✅ Step 3 terminé — Lance step4_benchmark.py\n')


if __name__ == '__main__':
    main()
