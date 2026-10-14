"""Verify saved metrics and their aggregation without retraining."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ids.core import config, digest, object_hash, read_json, source_hash, verify


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_tables(folder, model_column='model', prefix=''):
    runs = pd.read_csv(folder / (prefix + 'per_run.csv'))
    summary = pd.read_csv(folder / (prefix + 'mean_sd.csv'))
    keys = [model_column, 'percentile', 'policy']
    require(not runs.duplicated(keys + ['seed']).any(), 'Duplirano pokretanje')
    groups = runs.groupby(keys, dropna=False)
    require(len(groups) == len(summary), 'Broj zbirnih redova se ne podudara')
    checks = 0
    for _, row in summary.iterrows():
        key = tuple(row[k] for k in keys)
        group = groups.get_group(key)
        require(len(group) == row['n_runs'], 'Pogrešan broj pokretanja')
        if len(group) != 1:
            require(sorted(group.seed.tolist()) == config()['seeds'], 'Nedostaje seed')
        else:
            require(str(row[model_column]).startswith('pca'), 'Neočekivan pojedinačni model')
        for name in summary.columns:
            if name.endswith('_mean') or name.endswith('_sd'):
                metric, kind = name.rsplit('_', 1)
                values = group[metric]
                expected = values.mean() if kind == 'mean' else values.std(ddof=1)
                require(np.isclose(expected, row[name], rtol=1e-10, atol=1e-10, equal_nan=True),
                        f'Neusaglašena agregacija: {folder.name}, {key}, {name}')
                checks += 1
    return checks


def main():
    results = ROOT / 'results'
    decision = read_json(results / 'decision.json')
    provenance = read_json(ROOT / 'evidence/provenance.json')
    recorded = provenance['recorded_experiment']
    distributed = provenance['distribution']
    require(source_hash() == distributed['code_hash'], 'Distributed code fingerprint mismatch')
    require(object_hash(config()) == distributed['config_hash'], 'Distributed configuration mismatch')
    require(decision['code_hash'] == recorded['code_hash'], 'Recorded code fingerprint mismatch')
    require(decision['config_hash'] == recorded['config_hash'], 'Recorded configuration mismatch')
    require(digest(results / 'decision.json') == distributed['decision_sha256'], 'Distributed decision mismatch')
    for rel, expected in distributed['module_sha256'].items():
        require(digest(ROOT / rel) == expected, f'Module fingerprint mismatch: {rel}')
    for rel, hashes in distributed['normalized_metadata'].items():
        require(digest(ROOT / rel) == hashes['distributed_sha256'], f'Metadata fingerprint mismatch: {rel}')
    require(decision['main_data_hash'] == distributed['normalized_metadata'][
        'evidence/data/cic2017/data_identity.json']['recorded_sha256'], 'Recorded data identity mismatch')
    require(digest(results / 'decision.json') == read_json(results / 'decision.sha256.json')['sha256'],
            'Odluka je promenjena')
    require((decision['architecture'], decision['percentile_ae'], decision['percentile_if']) ==
            ('ae_l8', 98, 95), 'Neočekivana radna tačka')
    checks = check_tables(results / 'development', 'architecture', 'development_')
    areas = [('summary_final_test', 909632, 326853),
             ('external/summary_external_test', 5410088, 31050),
             ('external/summary_external_diagnostic', 287551, 94197)]
    for rel, total, positive in areas:
        folder = results / rel
        verify(folder)
        require(read_json(folder / 'selected_operating_points.json')['decision_hash'] ==
                recorded['decision_sha256'], 'Recorded result decision mismatch')
        checks += check_tables(folder)
        frame = pd.read_csv(folder / 'per_run.csv')
        primary = frame[frame.policy == 'primary']
        require((primary[['tn', 'fp', 'fn', 'tp']].sum(axis=1) == total).all(), 'Pogrešan broj tokova')
        require(((primary.fn + primary.tp) == positive).all(), 'Pogrešan broj pozitivnih tokova')
    for rel in ['figures_final_test', 'external/figures_external_test']:
        verify(results / rel)
        require(read_json(results / rel / 'figure_context.json')['decision_hash'] ==
                recorded['decision_sha256'], 'Recorded figure decision mismatch')
    print(f'Provera uspešna: kod, konfiguracija, odluka, kontrolne sume i {checks} agregiranih vrednosti.')
    print('Provera obuhvata sačuvane zbirne rezultate; modeli i pojedinačni skorovi nisu u paketu.')


if __name__ == '__main__':
    main()
