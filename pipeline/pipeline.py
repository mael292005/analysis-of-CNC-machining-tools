"""
pipeline.py
───────────
Point d'entrée unique — lance les 5 étapes dans l'ordre.

Usage :
    python pipeline.py --dataset ../dataset_4/phm2010_features.csv
    python pipeline.py --dataset ../dataset_2/Exp2.csv --cv 10
    python pipeline.py --dataset ../dataset_4/phm2010_features.csv --from_step 3
    python pipeline.py --dataset ../dataset_4/phm2010_features.csv --only_step 5

Steps :
    1  profiler          — analyse le dataset, produit dataset_profile.json
    2  feature_engine    — nettoie/extrait features → X_clean.csv, y_clean.csv
    3  balancer          — équilibre les classes → X_balanced.csv, y_balanced.csv
    4  benchmark         — 10 modèles OOF CV → benchmark_results.json
    5  report            — graphes + rapport HTML
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SCRIPT_FOLDER = os.path.basename(SCRIPT_DIR).lower()
SUBFOLDER_NAMES = ('pipeline', 'scripts', 'src', 'code', 'pipeline_cnc')
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR) if SCRIPT_FOLDER in SUBFOLDER_NAMES else SCRIPT_DIR

STEPS = {
    1: ('step1_profiler.py',       'Profiling du dataset'),
    2: ('step2_feature_engine.py', 'Feature engineering & nettoyage'),
    3: ('step3_balancer.py',       'Équilibrage des classes'),
    4: ('step4_benchmark.py',      'Benchmark 10 modèles ML'),
    5: ('step5_report.py',         'Génération du rapport'),
}


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    for base in (PROJECT_ROOT, SCRIPT_DIR):
        c = os.path.join(base, path)
        if os.path.exists(c):
            return c
    return path


def run_step(step_num: int, extra_args: list = None) -> bool:
    script_name, description = STEPS[step_num]
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.isfile(script_path):
        print(f'  [ERROR] Script introuvable : {script_path}')
        return False

    W = 62
    print(f'\n{"═" * W}')
    print(f'  STEP {step_num}/5 — {description.upper()}')
    print(f'{"═" * W}')

    cmd = [sys.executable, script_path] + (extra_args or [])
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)

    if result.returncode != 0:
        print(f'\n  ❌ Step {step_num} échoué (code {result.returncode})')
        return False

    return True


def banner():
    W = 62
    print(f'\n{"█" * W}')
    print(f'  PIPELINE CNC — PRÉDICTION USURE OUTIL')
    print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"█" * W}')


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline complète de prédiction usure outil CNC'
    )
    parser.add_argument('--dataset',    type=str, required=True,
                        help='Chemin vers le CSV ou dossier dataset')
    parser.add_argument('--cv',         type=int, default=5,
                        help='Nombre de folds CV (défaut: 5)')
    parser.add_argument('--strategy',   type=str, default='auto',
                        choices=['auto', 'smote', 'weights', 'none'],
                        help='Stratégie d\'équilibrage (défaut: auto)')
    parser.add_argument('--from_step',  type=int, default=1,
                        choices=[1, 2, 3, 4, 5],
                        help='Reprendre depuis cette étape (défaut: 1)')
    parser.add_argument('--only_step',  type=int, default=None,
                        choices=[1, 2, 3, 4, 5],
                        help='Lancer uniquement cette étape')
    args = parser.parse_args()

    # Résolution du chemin dataset
    dataset_path = resolve_path(args.dataset)
    if not os.path.exists(dataset_path):
        print(f'[ERROR] Dataset introuvable : {args.dataset}')
        print(f'        Testé aussi dans    : {PROJECT_ROOT}')
        sys.exit(1)

    banner()
    print(f'\n  Dataset  : {dataset_path}')
    print(f'  CV folds : {args.cv}')
    print(f'  Balance  : {args.strategy}')

    # Détermine quelles étapes lancer
    if args.only_step:
        steps_to_run = [args.only_step]
    else:
        steps_to_run = list(range(args.from_step, 6))

    print(f'  Étapes   : {steps_to_run}')

    # Arguments spécifiques à chaque step
    step_args = {
        1: ['--dataset', dataset_path],
        2: [],
        3: ['--strategy', args.strategy],
        4: ['--cv', str(args.cv)],
        5: [],
    }

    t_start = datetime.now()
    failed  = False

    for step_num in steps_to_run:
        success = run_step(step_num, step_args.get(step_num, []))
        if not success:
            print(f'\n  Pipeline interrompue à l\'étape {step_num}.')
            failed = True
            break

    elapsed = (datetime.now() - t_start).total_seconds()
    W = 62

    if not failed:
        print(f'\n{"█" * W}')
        print(f'  ✅ PIPELINE TERMINÉE en {elapsed:.0f}s')
        report = os.path.join(SCRIPT_DIR, 'report.html')
        if os.path.isfile(report):
            print(f'  📊 Rapport → {report}')
        print(f'{"█" * W}\n')
    else:
        print(f'\n{"█" * W}')
        print(f'  ❌ Pipeline échouée après {elapsed:.0f}s')
        print(f'{"█" * W}\n')
        sys.exit(1)


if __name__ == '__main__':
    main()
