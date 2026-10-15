"""Does each reference-free coherence axis separate the oracle from random output?

Runs the CPU-only baselines over a whole split and reports the distribution of
every primary axis. An axis on which `random_valid` matches `oracle` does not
discriminate, and no amount of model inference will make it do so.

This is a diagnostic: it calls the campaign's own `run_model` baselines and
`run_coherence`, writes nothing into a campaign directory, and is not evidence
for admission or for the dissertation's tables.

Usage:
  python scripts/axis_discrimination.py --dataset-root data/output_carla400_protocol10 \
      --out-dir /tmp/axis_check/carla [--backends oracle,random_valid] [--limit 25]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.causal_coherence import run_coherence
from analysis.run_model import run_model

AXES = ('monotonicity', 'responsiveness', 'locality', 'action_direction',
        'speed_response', 'speed_response_brake', 'speed_response_accelerate')
DEFAULT_BACKENDS = ('oracle', 'action_profile_kinematic', 'random_valid')


def per_scenario_values(coherence: dict) -> dict:
    out = {a: [] for a in AXES}
    for row in coherence.get('per_scenario', []):
        for a in AXES:
            if row.get(a) is not None:
                out[a].append(round(float(row[a]), 3))
    return out


def run(args):
    root = Path(args.dataset_root).resolve()
    scenarios = sorted(p.name for p in root.glob('scenario_*') if p.is_dir())
    if args.limit:
        scenarios = scenarios[:args.limit]
    if not scenarios:
        raise SystemExit(f'no scenario_* directories under {root}')
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f'{root.name}: {len(scenarios)} scenarios, backends {", ".join(args.backends)}\n')

    table = {}
    for backend in args.backends:
        cell = out / backend
        t0 = time.time()
        run_model(str(root), str(cell), backend=backend, condition='video',
                  seed=0, temperature=0.0, skip_existing=True,
                  scenario_names=scenarios)
        coherence = run_coherence(str(root), str(cell))
        json.dump(coherence, open(cell / 'coherence.json', 'w'), indent=1, sort_keys=True)
        table[backend] = per_scenario_values(coherence)
        print(f'  {backend:26s} done in {time.time() - t0:6.1f}s')

    print(f'\n{"=" * 96}\nPER-AXIS DISTRIBUTION over {len(scenarios)} scenarios\n{"=" * 96}')
    for axis in AXES:
        print(f'\n{axis}')
        print('  %-26s %4s %7s %7s %10s %8s'
              % ('backend', 'n', 'mean', 'median', 'at 0 or 1', 'distinct'))
        for backend in args.backends:
            v = table[backend][axis]
            if not v:
                print(f'  {backend:26s} {0:>4d}       --      --          --        --')
                continue
            edge = sum(1 for x in v if x in (0.0, 1.0))
            print(f'  {backend:26s} {len(v):>4d} {statistics.fmean(v):7.3f} '
                  f'{statistics.median(v):7.3f} {100 * edge / len(v):9.0f}% {len(set(v)):8d}')
        o, r = table[args.backends[0]][axis], table[args.backends[-1]][axis]
        if o and r:
            gap = statistics.fmean(o) - statistics.fmean(r)
            verdict = 'SEPARATES' if gap > 0.15 else 'DOES NOT SEPARATE'
            print(f'  -> {args.backends[0]} minus {args.backends[-1]} = {gap:+.3f}   {verdict}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--backends', default=','.join(DEFAULT_BACKENDS))
    p.add_argument('--limit', type=int, help='first N scenarios only (timing trial)')
    a = p.parse_args()
    a.backends = [b.strip() for b in a.backends.split(',') if b.strip()]
    run(a)


if __name__ == '__main__':
    main()
