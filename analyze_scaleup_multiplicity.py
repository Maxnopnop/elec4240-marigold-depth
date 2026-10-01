"""Retrospective multiplicity sensitivity for all 21 previously published contrasts."""
from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
from run_expanded import read, write, sha
from robustness_stats import bootstrap, holm


def analyze(destination):
    old = Path('results/scaleup_v2')
    batches = defaultdict(list)
    sources = {}
    for path in sorted((old/'evaluations'/'test').glob('*/per_image.json')):
        rows = read(path)
        first = rows[0]
        group = 'expert' if first['mode'] == 'expert' else f"{first['mode']}_r{first['resolution']}"
        batches[group].append(rows)
        sources[path.as_posix()] = sha(path)
    ids = [r['id'] for r in batches['base_r256'][0]]
    averages = {}
    for group, runs in batches.items():
        assert all([r['id'] for r in rows] == ids for rows in runs)
        averages[group] = np.mean([[r['abs_rel'] for r in rows] for rows in runs], axis=0)
    with (old/'paired_comparisons.csv').open(newline='', encoding='utf-8') as f:
        original = list(csv.DictReader(f))
    assert len(original) == 21
    comparisons = []
    for prior in original:
        c, r = prior['compared'], prior['reference']
        delta = averages[c]-averages[r]
        assert abs(delta.mean()-float(prior['abs_rel_difference'])) < 1e-12
        result = bootstrap(delta[None, None, :], seed=4265, family=21)['scene']
        comparisons.append({'compared': c, 'reference': r, **result})
    for result, p in zip(comparisons, holm([r['p_bootstrap'] for r in comparisons])):
        result['p_holm'] = p
    output = {'analysis': 'post-hoc scene-only sensitivity; fixed trained models, 21-comparison family',
              'source_sha256': sources, 'replicates': 20000, 'comparisons': comparisons}
    write(destination, output)
    return output


if __name__ == '__main__':
    result = analyze(Path('results/robustness_v3/scaleup_posthoc_multiplicity.json'))
    for r in result['comparisons']:
        print(r['compared'], '-', r['reference'], 'delta', round(r['difference'], 6), 'Holm p', round(r['p_holm'], 4))
