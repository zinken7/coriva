"""Generate and score one non-learning program's cells under the panel's baselines root.

    .venv/bin/python scripts/run_baseline_program.py action_profile_typical [--root results/api_panel/baselines]

The programs read no prompt, so the cells live under one root shared by every arm. The cell layout,
campaign identity and scoring are those of `scripts/run_api_panel.py` (`panel_cells`) and
`analysis.evaluation_campaign.score_cell`.
"""
import argparse, json, sys
from pathlib import Path

_here = Path(__file__).resolve()
_root = next((p for p in _here.parents if (p / "analysis").is_dir()), _here.parent)
sys.path.insert(0, str(_root))

from analysis.evaluation_campaign import BASELINES, score_cell, write_campaign_identity  # noqa: E402
from analysis.run_model import run_model  # noqa: E402
from analysis.evaluation_campaign import sha, write_json  # noqa: E402
from analysis.evaluation_report import collect_cell  # noqa: E402

import re as _re
_SPLITS = eval('{' + _re.search(r'SPLITS = \{(.*?)\}', (_root / 'scripts' / 'run_api_panel.py').read_text(), _re.S).group(1) + '}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("program", choices=list(BASELINES))
    ap.add_argument("--root", type=Path, default=_root / "results" / "api_panel" / "baselines")
    ap.add_argument("--splits", nargs="*", default=["carla", "nuscenes"])
    args = ap.parse_args()
    root = args.root.resolve()
    frozen = json.loads((root / "frozen.json").read_text())
    campaign = {"campaign_dir": str(root), "splits": {s: {"path": str(root / "inputs" / s)} for s in args.splits}}
    for split in args.splits:
        cell = dict(split=split, backend=args.program, condition="video", configuration=None,
                    out_dir=str(root / "cells" / split / args.program))
        Path(cell["out_dir"]).mkdir(parents=True, exist_ok=True)
        write_campaign_identity(campaign, cell)
        stats = run_model(campaign["splits"][split]["path"], cell["out_dir"], backend=args.program,
                          condition="video", seed=0, skip_existing=True)
        print(f"[{args.program} {split}] generated: {stats if isinstance(stats, dict) else ''}")
        score_cell(campaign, cell, frozen)
        # the arm report reads the per-cell metrics file that scripts/run_api_panel.py writes after scoring
        data = collect_cell(cell, frozen["datasets"][split], split_path=str((_root / _SPLITS[split]).resolve()),
                            frozen_sha256=sha(root / "frozen.json"))
        write_json(Path(cell["out_dir"]) / "collected.json", data)
        rep = json.loads((Path(cell["out_dir"]) / "report.json").read_text())["summary"]
        print(f"[{args.program} {split}] CTMS {rep.get('CTMS_mean'):.3f}  pair valid {rep.get('L4_pair_valid_rate'):.2f}")


if __name__ == "__main__":
    main()
