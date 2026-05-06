"""
step4_benchmark.py
──────────────────
Lit X_balanced.csv + y_balanced.csv + balancer_info.json (Step 3)
et lance les 10 algorithmes en cross-validation OOF stricte.

Intègre les class_weights du Step 3 pour les modèles compatibles.
Sauvegarde benchmark_results.json + benchmark_summary.csv.

Usage :
    python step4_benchmark.py
    python step4_benchmark.py --cv 10
    python step4_benchmark.py --profile dataset_profile.json --cv 5
"""

import os
import sys
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
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FOLDER = os.path.basename(SCRIPT_DIR).lower()
SUBFOLDER_NAMES = ('pipeline', 'scripts', 'src', 'code', 'pipeline_cnc')
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR) if SCRIPT_FOLDER in SUBFOLDER_NAMES else SCRIPT_DIR
RANDOM_STATE  = 42


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    for base in (PROJECT_ROOT, SCRIPT_DIR):
        c = os.path.join(base, path)
        if os.path.exists(c):
            return c
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  MODÈLES — avec support class_weight
# ─────────────────────────────────────────────────────────────────────────────
def get_models(task: str, class_weights: dict | None) -> dict:
    """
    Construit les 10 modèles.
    Si class_weights est fourni, l'injecte dans les modèles compatibles.
    """
    # sklearn attend {classe: poids} avec les types originaux
    cw_sklearn = None
    if class_weights:
        # Essaie de convertir les clés en int si possible, sinon garde str
        cw_sklearn = {}
        for k, v in class_weights.items():
            try:
                cw_sklearn[int(k)] = v
            except (ValueError, TypeError):
                cw_sklearn[k] = v

    cw = cw_sklearn  # alias court

    if task == 'classification':
        return {
            '01_RandomForest':          RandomForestClassifier(
                n_estimators=300, max_depth=None, class_weight=cw,
                random_state=RANDOM_STATE, n_jobs=-1),
            '02_ExtraTrees':            ExtraTreesClassifier(
                n_estimators=300, class_weight=cw,
                random_state=RANDOM_STATE, n_jobs=-1),
            '03_GradientBoosting':      GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                random_state=RANDOM_STATE),
            '04_HistGradientBoosting':  HistGradientBoostingClassifier(
                max_iter=200, class_weight=cw,
                random_state=RANDOM_STATE),
            '05_Bagging':               BaggingClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            '06_SVM':                   Pipeline([
                ('sc', StandardScaler()),
                ('m',  SVC(kernel='rbf', C=10, class_weight=cw,
                           probability=True, random_state=RANDOM_STATE))]),
            '07_KNN':                   Pipeline([
                ('sc', StandardScaler()),
                ('m',  KNeighborsClassifier(n_neighbors=5, n_jobs=-1))]),
            '08_DecisionTree':          DecisionTreeClassifier(
                max_depth=10, class_weight=cw,
                random_state=RANDOM_STATE),
            '09_LogisticRegression':    Pipeline([
                ('sc', StandardScaler()),
                ('m',  LogisticRegression(
                    max_iter=2000, C=1.0, class_weight=cw,
                    random_state=RANDOM_STATE))]),
            '10_MLP':                   Pipeline([
                ('sc', StandardScaler()),
                ('m',  MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=500,
                    early_stopping=True, random_state=RANDOM_STATE))]),
        }
    else:
        return {
            '01_RandomForest':          RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            '02_ExtraTrees':            ExtraTreesRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            '03_GradientBoosting':      GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                random_state=RANDOM_STATE),
            '04_HistGradientBoosting':  HistGradientBoostingRegressor(
                max_iter=200, random_state=RANDOM_STATE),
            '05_Bagging':               BaggingRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            '06_SVR':                   Pipeline([
                ('sc', StandardScaler()),
                ('m',  SVR(kernel='rbf', C=10, epsilon=0.01))]),
            '07_KNN':                   Pipeline([
                ('sc', StandardScaler()),
                ('m',  KNeighborsRegressor(n_neighbors=5, n_jobs=-1))]),
            '08_DecisionTree':          DecisionTreeRegressor(
                max_depth=10, random_state=RANDOM_STATE),
            '09_Ridge':                 Pipeline([
                ('sc', StandardScaler()),
                ('m',  Ridge(alpha=1.0))]),
            '10_MLP':                   Pipeline([
                ('sc', StandardScaler()),
                ('m',  MLPRegressor(
                    hidden_layer_sizes=(128, 64), max_iter=500,
                    early_stopping=True, random_state=RANDOM_STATE))]),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  BENCHMARK OOF
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(X, y, task, models, n_splits):
    if task == 'classification':
        cv      = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=RANDOM_STATE)
        scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
    else:
        cv      = KFold(n_splits=n_splits, shuffle=True,
                        random_state=RANDOM_STATE)
        scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']

    metrics_dict = {}
    preds_dict   = {}
    total        = len(models)

    for i, (name, model) in enumerate(models.items(), 1):
        short = name.split('_', 1)[1]
        print(f'  [{i:02d}/{total}] {short:<28}', end=' ', flush=True)
        t0 = time.time()

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
                      f"BalAcc={m['balanced_accuracy']:.4f}  {elapsed:.1f}s ✓")
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
                print(f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                      f"R²={m['r2']:.4f}±{m['r2_std']:.3f}  {elapsed:.1f}s ✓")

        except Exception as e:
            elapsed = time.time() - t0
            m = {'status': 'error', 'error': str(e), 'fit_time_s': round(elapsed, 2)}
            print(f'ERREUR — {e}')
            metrics_dict[name] = m
            preds_dict[name]   = {'error': str(e)}
            continue

        metrics_dict[name] = m

        # Prédictions OOF — aucun point prédit par son propre modèle
        try:
            y_oof = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            preds_dict[name] = {
                'y_true': y.tolist(),
                'y_pred': y_oof.tolist(),
                'oof'   : True,
            }
        except Exception as e:
            preds_dict[name] = {'error': f'oof failed: {e}'}

    return metrics_dict, preds_dict


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Step 4 — Benchmark 10 modèles ML (OOF CV)'
    )
    parser.add_argument('--cv',      type=int, default=5)
    parser.add_argument('--profile', type=str, default='dataset_profile.json')
    args = parser.parse_args()

    profile_path   = resolve_path(args.profile)
    x_path         = resolve_path('X_balanced.csv')
    y_path         = resolve_path('y_balanced.csv')
    balancer_path  = resolve_path('balancer_info.json')

    for p, name in [(profile_path, 'dataset_profile.json'),
                    (x_path, 'X_balanced.csv'),
                    (y_path, 'y_balanced.csv')]:
        if not os.path.isfile(p):
            print(f'[ERROR] Fichier manquant : {name}')
            print('        Lance d\'abord step3_balancer.py')
            sys.exit(1)

    with open(profile_path, encoding='utf-8') as f:
        profile = json.load(f)

    class_weights = None
    if os.path.isfile(balancer_path):
        with open(balancer_path, encoding='utf-8') as f:
            balancer_info = json.load(f)
        class_weights = balancer_info.get('class_weights')

    print(f'\n{"═" * 62}')
    print(f'  STEP 4 — BENCHMARK ML')
    print(f'  Dataset : {profile["filename"]}')
    print(f'  Tâche   : {profile["task"].upper()}')
    if class_weights:
        print(f'  Weights : {class_weights}')
    print(f'{"═" * 62}\n')

    X   = pd.read_csv(x_path)
    y_s = pd.read_csv(y_path)['target']
    task = profile.get('task', 'classification')

    # Encodage si classification
    le = None
    if task == 'classification':
        le = LabelEncoder()
        y  = le.fit_transform(y_s)
        print(f'  Classes : {dict(zip(le.classes_, range(len(le.classes_))))}')
        # Recalcule les weights avec les indices encodés
        if class_weights:
            class_weights = {
                int(le.transform([k])[0]) if k in le.classes_ else k: v
                for k, v in class_weights.items()
            }
    else:
        y = y_s.values.astype(float)

    # Ajuste le nb de folds à la classe minoritaire
    if task == 'classification':
        min_cls = int(np.bincount(y).min())
        n_splits = min(args.cv, min_cls)
    else:
        n_splits = min(args.cv, len(X))
    n_splits = max(2, n_splits)

    if n_splits < args.cv:
        print(f'  ⚠  CV réduit à {n_splits} folds (classe min. = {min_cls} ex.)')

    models = get_models(task, class_weights)

    print(f'  {len(models)} modèles | {n_splits}-Fold OOF CV\n')
    metrics, preds = run_benchmark(X.values, y, task, models, n_splits)

    # Décodage labels
    if le is not None:
        for name, pd_ in preds.items():
            if 'error' not in pd_:
                pd_['y_true']  = le.inverse_transform([int(v) for v in pd_['y_true']]).tolist()
                pd_['y_pred']  = le.inverse_transform([int(v) for v in pd_['y_pred']]).tolist()
                pd_['classes'] = le.classes_.tolist()

    # Classement
    print(f'\n{"─" * 62}')
    print('  CLASSEMENT FINAL')
    print(f'{"─" * 62}')
    ok     = {k: v for k, v in metrics.items() if v.get('status') == 'ok'}
    medals = {1: '🥇', 2: '🥈', 3: '🥉'}
    if task == 'classification':
        ranked = sorted(ok.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
        for rank, (name, m) in enumerate(ranked, 1):
            short = name.split('_', 1)[1]
            print(f'  {medals.get(rank,"  ")} #{rank}  {short:<28}  '
                  f"F1={m['f1_macro']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  "
                  f"Acc={m['accuracy']:.4f}")
    else:
        ranked = sorted(ok.items(), key=lambda x: x[1]['r2'], reverse=True)
        for rank, (name, m) in enumerate(ranked, 1):
            short = name.split('_', 1)[1]
            print(f'  {medals.get(rank,"  ")} #{rank}  {short:<28}  '
                  f"R²={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    # Sauvegarde
    result = {
        'meta': {
            'dataset'     : profile.get('filename'),
            'target'      : profile.get('target_column'),
            'task'        : task,
            'features'    : list(X.columns),
            'n_samples'   : int(len(X)),
            'n_features'  : int(X.shape[1]),
            'cv_folds'    : n_splits,
            'cv_method'   : 'cross_val_predict OOF',
            'class_weights': class_weights,
            'timestamp'   : datetime.now().isoformat(),
        },
        'metrics'    : metrics,
        'predictions': preds,
    }

    res_path = os.path.join(SCRIPT_DIR, 'benchmark_results.json')
    with open(res_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    rows = [{'model': k.split('_', 1)[1], **v} for k, v in metrics.items()]
    csv_path = os.path.join(SCRIPT_DIR, 'benchmark_summary.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    profile['step4_done']    = True
    profile['step4_best']    = ranked[0][0].split('_', 1)[1] if ranked else None
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, default=str, ensure_ascii=False)

    print(f'\n  💾 benchmark_results.json → {res_path}')
    print(f'  💾 benchmark_summary.csv  → {csv_path}')
    print(f'  ✅ Step 4 terminé — Lance step5_report.py\n')


if __name__ == '__main__':
    main()
