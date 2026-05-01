"""
benchmark_tool_wear.py
──────────────────────
Pipeline de benchmark ML — Prédiction usure outil CNC.

Lance le script, il te guide interactivement pour :
  1. Choisir le fichier CSV
  2. Valider / corriger la colonne cible
  3. Valider / corriger les features (avec détection leakage)
  4. Lance les 10 algos en cross-validation OOF stricte
  5. Sauvegarde benchmark_results.json + benchmark_summary.csv

Usage :
    python benchmark_tool_wear.py
    python benchmark_tool_wear.py --output ./results/
    python benchmark_tool_wear.py --cv 10
"""

import os
import sys
import glob
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (
    StratifiedKFold, KFold,
    cross_val_predict, cross_validate,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
WEAR_KEYWORDS = [
    'tcond', 'tool_cond', 'tool_state', 'wear', 'vbb', 'vb_mean',
    'usure', 'flank', 'toolwear', 'condition', 'etat',
    'label', 'class', 'target', 'output',
]
N_SPLITS               = 5
RANDOM_STATE           = 42
LEAKAGE_CORR_THRESHOLD = 0.98


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS UI
# ─────────────────────────────────────────────────────────────────────────────
def banner(text: str, char: str = '─', width: int = 62) -> None:
    print(f'\n{char * width}')
    for line in text.splitlines():
        print(f'  {line}')
    print(f'{char * width}')


def ask(prompt: str, default: str = '') -> str:
    hint = f' [{default}]' if default else ''
    val  = input(f'{prompt}{hint} : ').strip()
    return val if val else default


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = '[O/n]' if default else '[o/N]'
    val  = input(f'{prompt} {hint} : ').strip().lower()
    if val == '':
        return default
    return val in ('o', 'oui', 'y', 'yes', '1')


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 1 — SÉLECTION DU FICHIER
# ─────────────────────────────────────────────────────────────────────────────
def select_dataset():
    """
    Liste les CSV trouvés dans le répertoire courant (jusqu'à 2 niveaux).
    L'utilisateur choisit par numéro ou tape un chemin complet.
    Répète jusqu'à succès.
    """
    banner('ÉTAPE 1 — SÉLECTION DU DATASET', '═')

    candidates = sorted(set(
        glob.glob('**/*.csv', recursive=True) + glob.glob('*.csv')
    ))

    if candidates:
        print('\n  CSV détectés automatiquement :')
        for i, path in enumerate(candidates, 1):
            size_kb = os.path.getsize(path) / 1024
            print(f'    [{i:2d}] {path}  ({size_kb:.0f} KB)')
    else:
        print('\n  Aucun CSV trouvé dans le répertoire courant.')

    print('\n  → Entrer un numéro de la liste, ou le chemin complet vers le CSV')
    print('  → Ctrl+C pour quitter\n')

    while True:
        raw = input('  Choix : ').strip()
        if not raw:
            print('  ⚠  Entrée vide, réessaie.')
            continue

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(candidates):
                path = candidates[idx]
            else:
                print(f'  ⚠  Numéro invalide (1-{len(candidates)})')
                continue
        else:
            path = raw

        if not os.path.isfile(path):
            print(f'  ⚠  Fichier introuvable : {path}')
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f'  ⚠  Erreur lecture : {e}')
            continue

        print(f'\n  ✅ Chargé : {path}')
        print(f'     {df.shape[0]} lignes × {df.shape[1]} colonnes')
        print(f'     Colonnes : {list(df.columns)}\n')
        return df, path


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 2 — SÉLECTION DE LA COLONNE CIBLE
# ─────────────────────────────────────────────────────────────────────────────
def auto_detect_target(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in WEAR_KEYWORDS:
        if kw in cols_lower:
            return cols_lower[kw]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[-1] if num_cols else None


def auto_detect_task(series: pd.Series) -> str:
    if series.dtype == object:
        return 'classification'
    n_unique = series.nunique()
    ratio    = n_unique / max(len(series), 1)
    if n_unique <= 15 or ratio < 0.05:
        return 'classification'
    return 'regression'


def select_target_column(df: pd.DataFrame):
    """
    Affiche toutes les colonnes avec leurs stats.
    Propose une suggestion auto.
    L'utilisateur confirme ou choisit une autre.
    Boucle jusqu'à validation explicite.
    """
    banner('ÉTAPE 2 — COLONNE CIBLE (ce qu\'on veut prédire)', '═')

    suggestion = auto_detect_target(df)

    while True:
        # Affichage du tableau des colonnes
        print(f'\n  {"#":>3}  {"Nom":<28}  {"Type":<12}  {"N uniques":>10}  {"Exemple":<20}  Note')
        print(f'  {"─"*3}  {"─"*28}  {"─"*12}  {"─"*10}  {"─"*20}  {"─"*15}')
        for i, col in enumerate(df.columns, 1):
            dtype   = str(df[col].dtype)
            n_uniq  = df[col].nunique()
            example = str(df[col].dropna().iloc[0])[:20] if not df[col].dropna().empty else 'N/A'
            note    = '◄ suggestion' if col == suggestion else ''
            print(f'  {i:>3}  {col:<28}  {dtype:<12}  {n_uniq:>10}  {example:<20}  {note}')

        print()
        raw = ask(f'  Colonne cible (numéro ou nom exact)', default=suggestion or '')

        # Résolution
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(df.columns):
                target_col = df.columns[idx]
            else:
                print(f'  ⚠  Numéro invalide (1-{len(df.columns)}), réessaie.')
                continue
        elif raw in df.columns:
            target_col = raw
        else:
            print(f'  ⚠  "{raw}" non trouvé dans les colonnes.')
            continue

        # Infos + détection tâche
        series     = df[target_col].dropna()
        n_unique   = series.nunique()
        task_guess = auto_detect_task(series)

        print(f'\n  → Colonne sélectionnée : "{target_col}"')
        print(f'     Valeurs uniques ({n_unique}) : {sorted(series.unique())[:20]}')
        print(f'     Tâche auto-détectée : {task_guess.upper()}')

        # Confirmation tâche
        if task_guess == 'classification':
            ok_task = ask_yes_no('  → Confirmer CLASSIFICATION', default=True)
            task = 'classification' if ok_task else 'regression'
        else:
            ok_task = ask_yes_no('  → Confirmer RÉGRESSION', default=True)
            task = 'regression' if ok_task else 'classification'

        # Confirmation colonne
        ok_col = ask_yes_no(f'\n  → Utiliser "{target_col}" comme cible ({task}) ?',
                             default=True)
        if ok_col:
            print(f'\n  ✅ Cible : "{target_col}" | Tâche : {task.upper()}')
            return target_col, task

        # Réinitialise la suggestion si l'user veut autre chose
        suggestion = None
        print('  → Sélection annulée, recommençons...')


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 3 — FEATURES + DÉTECTION LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────
def compute_correlation_with_target(df: pd.DataFrame, col: str,
                                    target_col: str, task: str) -> float:
    """Corrélation |r| entre une feature et la target (encodée si besoin)."""
    try:
        if task == 'classification' or df[target_col].dtype == object:
            le = LabelEncoder()
            y  = pd.Series(le.fit_transform(df[target_col].fillna('NA').astype(str)),
                           index=df.index)
        else:
            y = df[target_col]
        valid = df[[col]].join(y.rename('__target__')).dropna()
        if len(valid) < 5:
            return float('nan')
        return abs(valid[col].corr(valid['__target__']))
    except Exception:
        return float('nan')


def select_features(df: pd.DataFrame, target_col: str, task: str) -> list:
    """
    Affiche les features numériques avec corrélations.
    Détecte et signale le data leakage.
    L'utilisateur valide ou ajuste.
    """
    banner('ÉTAPE 3 — FEATURES + DÉTECTION LEAKAGE', '═')

    # Candidats : colonnes numériques hors target
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate = [c for c in num_cols if c != target_col]

    # Exclure quasi-constantes
    low_var = [c for c in candidate if df[c].std() < 1e-9]
    if low_var:
        print(f'\n  ⚠  Colonnes quasi-constantes exclues auto : {low_var}')
    candidate = [c for c in candidate if c not in low_var]

    if not candidate:
        print('  ⚠  Aucune feature numérique disponible. Vérifier le dataset.')
        sys.exit(1)

    # Calcul des corrélations
    print('\n  Calcul des corrélations avec la target...')
    corr_map = {}
    for col in candidate:
        corr_map[col] = compute_correlation_with_target(df, col, target_col, task)

    # Affichage
    print(f'\n  {"#":>3}  {"Nom":<30}  {"|r| target":>10}  Alerte')
    print(f'  {"─"*3}  {"─"*30}  {"─"*10}  {"─"*25}')
    for i, col in enumerate(candidate, 1):
        r     = corr_map[col]
        r_str = f'{r:.4f}' if not np.isnan(r) else '  N/A '
        alert = '⚠  LEAKAGE PROBABLE' if (not np.isnan(r) and r >= LEAKAGE_CORR_THRESHOLD) else ''
        print(f'  {i:>3}  {col:<30}  {r_str:>10}  {alert}')

    # Leakage auto-exclusion
    leaky = [c for c, r in corr_map.items()
             if not np.isnan(r) and r >= LEAKAGE_CORR_THRESHOLD]

    if leaky:
        print(f'\n  ┌─────────────────────────────────────────────────────')
        print(f'  │ ⚠  ATTENTION — DATA LEAKAGE DÉTECTÉ')
        print(f'  │ Les colonnes suivantes ont |r| ≥ {LEAKAGE_CORR_THRESHOLD} avec la target.')
        print(f'  │ Elles peuvent faire monter les scores à 99-100% de façon')
        print(f'  │ artificielle : le modèle "triche" en voyant quasi la réponse.')
        for c in leaky:
            print(f'  │   • {c}  (|r| = {corr_map[c]:.4f})')
        print(f'  └─────────────────────────────────────────────────────')
        exclude_leaky = ask_yes_no(
            '\n  → Exclure automatiquement ces colonnes suspectes ?', default=True
        )
        if exclude_leaky:
            candidate = [c for c in candidate if c not in leaky]
            print(f'  ✅ Colonnes exclues : {leaky}')
            if not candidate:
                print('  ⚠  Plus aucune feature après exclusion leakage.')
                print('     Relance avec --cv 5 et vérifie ton dataset.')
                sys.exit(1)
    else:
        print('\n  ✅ Aucun leakage détecté.')

    print(f'\n  Features retenues ({len(candidate)}) : {candidate}')

    # Modification manuelle optionnelle
    custom = ask_yes_no('\n  → Veux-tu modifier manuellement la liste ?', default=False)
    if custom:
        print('  → Entrer les NUMÉROS à EXCLURE (ex: 1 3), Entrée pour garder tout')
        raw = input('  Exclusions : ').strip()
        if raw:
            all_cands   = [c for c in num_cols if c != target_col and c not in low_var]
            excl_idx    = [int(x) - 1 for x in raw.split() if x.isdigit()]
            excl_cols   = [all_cands[i] for i in excl_idx if 0 <= i < len(all_cands)]
            candidate   = [c for c in candidate if c not in excl_cols]
            print(f'  ✅ Exclues manuellement : {excl_cols}')

    if not candidate:
        print('  ⚠  Aucune feature restante. Abandon.')
        sys.exit(1)

    print(f'\n  ✅ Features finales ({len(candidate)}) : {candidate}')
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
#  MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
def get_models(task: str) -> dict:
    if task == 'classification':
        return {
            '01_RandomForest':         RandomForestClassifier(
                                           n_estimators=200, max_depth=None,
                                           random_state=RANDOM_STATE, n_jobs=-1),
            '02_ExtraTrees':           ExtraTreesClassifier(
                                           n_estimators=200,
                                           random_state=RANDOM_STATE, n_jobs=-1),
            '03_GradientBoosting':     GradientBoostingClassifier(
                                           n_estimators=200, learning_rate=0.05,
                                           random_state=RANDOM_STATE),
            '04_HistGradientBoosting': HistGradientBoostingClassifier(
                                           max_iter=200, random_state=RANDOM_STATE),
            '05_Bagging':              BaggingClassifier(
                                           n_estimators=100,
                                           random_state=RANDOM_STATE, n_jobs=-1),
            '06_SVM':                  Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', SVC(kernel='rbf', C=10,
                                                     probability=True,
                                                     random_state=RANDOM_STATE))]),
            '07_KNN':                  Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', KNeighborsClassifier(
                                                     n_neighbors=5, n_jobs=-1))]),
            '08_DecisionTree':         DecisionTreeClassifier(
                                           max_depth=10, random_state=RANDOM_STATE),
            '09_LogisticRegression':   Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', LogisticRegression(
                                                     max_iter=2000, C=1.0,
                                                     random_state=RANDOM_STATE))]),
            '10_MLP':                  Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', MLPClassifier(
                                                     hidden_layer_sizes=(128, 64),
                                                     max_iter=500,
                                                     early_stopping=True,
                                                     random_state=RANDOM_STATE))]),
        }
    else:  # regression
        return {
            '01_RandomForest':         RandomForestRegressor(
                                           n_estimators=200,
                                           random_state=RANDOM_STATE, n_jobs=-1),
            '02_ExtraTrees':           ExtraTreesRegressor(
                                           n_estimators=200,
                                           random_state=RANDOM_STATE, n_jobs=-1),
            '03_GradientBoosting':     GradientBoostingRegressor(
                                           n_estimators=200, learning_rate=0.05,
                                           random_state=RANDOM_STATE),
            '04_HistGradientBoosting': HistGradientBoostingRegressor(
                                           max_iter=200, random_state=RANDOM_STATE),
            '05_Bagging':              BaggingRegressor(
                                           n_estimators=100,
                                           random_state=RANDOM_STATE, n_jobs=-1),
            '06_SVR':                  Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', SVR(kernel='rbf',
                                                     C=10, epsilon=0.01))]),
            '07_KNN':                  Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', KNeighborsRegressor(
                                                     n_neighbors=5, n_jobs=-1))]),
            '08_DecisionTree':         DecisionTreeRegressor(
                                           max_depth=10, random_state=RANDOM_STATE),
            '09_Ridge':                Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', Ridge(alpha=1.0))]),
            '10_MLP':                  Pipeline([
                                           ('sc', StandardScaler()),
                                           ('m', MLPRegressor(
                                                     hidden_layer_sizes=(128, 64),
                                                     max_iter=500,
                                                     early_stopping=True,
                                                     random_state=RANDOM_STATE))]),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK — Cross-Validation OOF stricte
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(X: np.ndarray, y: np.ndarray, task: str,
                  models: dict, n_splits: int) -> tuple:
    """
    Métriques via cross_validate (CV stricte).
    Prédictions via cross_val_predict (out-of-fold : chaque point est prédit
    par un modèle qui ne l'a JAMAIS vu à l'entraînement → 0 leakage).
    """
    if task == 'classification':
        cv      = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=RANDOM_STATE)
        scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
    else:
        cv      = KFold(n_splits=n_splits, shuffle=True,
                        random_state=RANDOM_STATE)
        scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']

    metrics_dict     = {}
    predictions_dict = {}
    total            = len(models)

    for i, (name, model) in enumerate(models.items(), 1):
        short_name = name.split('_', 1)[1]
        print(f'  [{i:02d}/{total}] {short_name:<28}', end=' ', flush=True)
        t0 = time.time()

        # ── Métriques via CV ──────────────────────────────────────────────
        try:
            cv_res  = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                     return_train_score=False,
                                     error_score='raise', n_jobs=1)
            elapsed = time.time() - t0

            if task == 'classification':
                m = {
                    'accuracy'          : round(float(cv_res['test_accuracy'].mean()), 4),
                    'accuracy_std'      : round(float(cv_res['test_accuracy'].std()),  4),
                    'balanced_accuracy' : round(float(cv_res['test_balanced_accuracy'].mean()), 4),
                    'f1_macro'          : round(float(cv_res['test_f1_macro'].mean()),    4),
                    'f1_macro_std'      : round(float(cv_res['test_f1_macro'].std()),     4),
                    'f1_weighted'       : round(float(cv_res['test_f1_weighted'].mean()), 4),
                    'fit_time_s'        : round(elapsed, 2),
                    'status'            : 'ok',
                }
                print(f"Acc={m['accuracy']:.4f}±{m['accuracy_std']:.3f}  "
                      f"F1={m['f1_macro']:.4f}±{m['f1_macro_std']:.3f}  "
                      f"{elapsed:.1f}s ✓")
            else:
                mae  = -cv_res['test_neg_mean_absolute_error'].mean()
                rmse = -cv_res['test_neg_root_mean_squared_error'].mean()
                r2   =  cv_res['test_r2'].mean()
                m = {
                    'mae'       : round(float(mae),  5),
                    'mae_std'   : round(float(cv_res['test_neg_mean_absolute_error'].std()), 5),
                    'rmse'      : round(float(rmse), 5),
                    'r2'        : round(float(r2),   4),
                    'r2_std'    : round(float(cv_res['test_r2'].std()), 4),
                    'fit_time_s': round(elapsed, 2),
                    'status'    : 'ok',
                }
                print(f"MAE={m['mae']:.4f}±{m['mae_std']:.4f}  "
                      f"RMSE={m['rmse']:.4f}  "
                      f"R²={m['r2']:.4f}±{m['r2_std']:.3f}  "
                      f"{elapsed:.1f}s ✓")

        except Exception as e:
            elapsed = time.time() - t0
            m = {'status': 'error', 'error': str(e), 'fit_time_s': round(elapsed, 2)}
            print(f'ERREUR — {e}')
            metrics_dict[name]     = m
            predictions_dict[name] = {'error': str(e)}
            continue

        metrics_dict[name] = m

        # ── Prédictions Out-of-Fold (cross_val_predict) ───────────────────
        # Chaque échantillon est prédit par un modèle entraîné SANS lui.
        # C'est la seule façon correcte d'obtenir des prédictions sur tout le
        # dataset sans risque de leakage train→test.
        try:
            y_pred_oof = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            predictions_dict[name] = {
                'y_true' : y.tolist(),
                'y_pred' : y_pred_oof.tolist(),
                'oof'    : True,  # flag de garantie : out-of-fold, sans leakage
            }
        except Exception as e:
            predictions_dict[name] = {'error': f'cross_val_predict failed: {e}'}

    return metrics_dict, predictions_dict


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Benchmark 10 modèles ML — Prédiction usure outil CNC'
    )
    parser.add_argument('--output', type=str, default='.',
                        help='Dossier de sortie (défaut: .)')
    parser.add_argument('--cv', type=int, default=N_SPLITS,
                        help=f'Nombre de folds CV (défaut: {N_SPLITS})')
    args = parser.parse_args()

    print('\n' + '═' * 62)
    print('  BENCHMARK TOOL WEAR — Pipeline ML CNC')
    print('  Prédictions out-of-fold : 0 leakage garanti')
    print('═' * 62)

    try:
        # ── Étapes interactives ───────────────────────────────────────────
        df, dataset_path = select_dataset()
        target_col, task = select_target_column(df)
        feature_cols     = select_features(df, target_col, task)

    except KeyboardInterrupt:
        print('\n\n  Abandon (Ctrl+C).')
        sys.exit(0)

    # ── Préparation X, y ──────────────────────────────────────────────────
    df_clean  = df[feature_cols + [target_col]].dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped:
        print(f'\n  ℹ  {n_dropped} ligne(s) avec NaN supprimées → {len(df_clean)} retenues')

    if len(df_clean) < 20:
        print(f'  ⚠  Seulement {len(df_clean)} lignes — résultats peu fiables.')

    X     = df_clean[feature_cols].values
    y_raw = df_clean[target_col].values

    label_encoder = None
    if task == 'classification':
        le            = LabelEncoder()
        y             = le.fit_transform(y_raw)
        label_encoder = le
        print(f'\n  Encodage classes : '
              + '  |  '.join(f'{c} → {le.transform([c])[0]}' for c in le.classes_))
    else:
        y = y_raw.astype(float)

    # Adapter le nombre de folds à la classe minoritaire (classification)
    if task == 'classification':
        min_class_count = int(np.bincount(y).min())
        n_splits_actual = min(args.cv, min_class_count)
    else:
        n_splits_actual = min(args.cv, len(df_clean))
    n_splits_actual = max(2, n_splits_actual)

    if n_splits_actual < args.cv:
        print(f'\n  ⚠  CV réduit à {n_splits_actual} folds '
              f'(classe minoritaire : {min_class_count} exemples)')

    # ── Benchmark ─────────────────────────────────────────────────────────
    models = get_models(task)
    banner(
        f'ÉTAPE 4 — BENCHMARK\n'
        f'{len(models)} modèles | {n_splits_actual}-Fold CV | OOF predictions',
        '═'
    )
    print()
    metrics, predictions = run_benchmark(X, y, task, models,
                                          n_splits=n_splits_actual)

    # ── Décoder labels ────────────────────────────────────────────────────
    if label_encoder is not None:
        for name, pdict in predictions.items():
            if 'error' not in pdict:
                pdict['y_true']  = label_encoder.inverse_transform(
                    [int(v) for v in pdict['y_true']]).tolist()
                pdict['y_pred']  = label_encoder.inverse_transform(
                    [int(v) for v in pdict['y_pred']]).tolist()
                pdict['classes'] = label_encoder.classes_.tolist()

    # ── Classement ────────────────────────────────────────────────────────
    banner('CLASSEMENT FINAL', '═')
    ok_models = {k: v for k, v in metrics.items() if v.get('status') == 'ok'}
    medals    = {1: '🥇', 2: '🥈', 3: '🥉'}

    if task == 'classification':
        ranked = sorted(ok_models.items(),
                        key=lambda x: x[1]['f1_macro'], reverse=True)
        for rank, (name, m) in enumerate(ranked, 1):
            short = name.split('_', 1)[1]
            print(f'  {medals.get(rank,"  ")} #{rank}  {short:<28}  '
                  f"Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
                  f"BalAcc={m['balanced_accuracy']:.4f}")
    else:
        ranked = sorted(ok_models.items(),
                        key=lambda x: x[1]['r2'], reverse=True)
        for rank, (name, m) in enumerate(ranked, 1):
            short = name.split('_', 1)[1]
            print(f'  {medals.get(rank,"  ")} #{rank}  {short:<28}  '
                  f"R²={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    # ── Sauvegarde ────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)

    output_data = {
        'meta': {
            'dataset'       : os.path.abspath(dataset_path),
            'dataset_name'  : os.path.basename(dataset_path),
            'target_column' : target_col,
            'task'          : task,
            'features'      : feature_cols,
            'n_samples'     : int(df_clean.shape[0]),
            'n_features'    : len(feature_cols),
            'cv_folds'      : n_splits_actual,
            'cv_method'     : 'cross_val_predict — out-of-fold (no leakage)',
            'timestamp'     : datetime.now().isoformat(),
        },
        'metrics'     : metrics,
        'predictions' : predictions,
    }

    json_path = os.path.join(args.output, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    rows = [{'model': k.split('_', 1)[1], **v} for k, v in metrics.items()]
    csv_path = os.path.join(args.output, 'benchmark_summary.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f'\n  💾 JSON  → {os.path.abspath(json_path)}')
    print(f'  💾 CSV   → {os.path.abspath(csv_path)}')
    print(f'\n  ✅ Terminé — Lance visualize_benchmark.ipynb pour les graphes.')
    print('═' * 62 + '\n')


if __name__ == '__main__':
    main()
