"""Step 1 of the pipeline: download -> normalise -> split -> write -> report.

    python scripts/prepare_data.py                 # all datasets
    python scripts/prepare_data.py --datasets musique qasper

Outputs
    data/processed/<ds>/{dev,test}.jsonl   unified Example records
    data/processed/<ds>/corpus.jsonl       every passage referenced by those examples
    data/splits/<ds>.json                  question ids per split (committed to git)
    reports/data_stats.{md,json}           descriptive statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfcheck.data.loaders import LOADERS  # noqa: E402
from selfcheck.data.sources import fetch  # noqa: E402
from selfcheck.data.splits import make_splits  # noqa: E402
from selfcheck.data.stats import split_stats, to_markdown  # noqa: E402


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raw, proc = ROOT / cfg["raw_dir"], ROOT / cfg["processed_dir"]
    splits_dir, reports = ROOT / cfg["splits_dir"], ROOT / cfg["reports_dir"]
    names = args.datasets or list(cfg["datasets"])

    fetch(raw, names)

    all_stats, pool_sizes = {}, {}
    for ds in names:
        dcfg = cfg["datasets"][ds]
        examples, corpus = LOADERS[ds](raw)
        pool_sizes[ds] = len(examples)
        splits = make_splits(ds, examples, dcfg["dev"], dcfg["test"], dcfg["stratify_by"], cfg["seed"])

        used = sorted({d for sp in splits.values() for e in sp for d in e.scope_doc_ids})
        write_jsonl(proc / ds / "corpus.jsonl", (corpus[d].to_dict() for d in used))
        for name, sp in splits.items():
            write_jsonl(proc / ds / f"{name}.jsonl", (e.to_dict() for e in sp))
        splits_dir.mkdir(parents=True, exist_ok=True)
        (splits_dir / f"{ds}.json").write_text(
            json.dumps({k: sorted(e.qid for e in v) for k, v in splits.items()}, indent=1)
        )
        all_stats[ds] = {"pool": split_stats(examples, corpus)} | {
            k: split_stats(v, corpus) for k, v in splits.items()
        }
        print(f"[{ds}] pool={len(examples)} dev={len(splits['dev'])} "
              f"test={len(splits['test'])} passages={len(used)}")

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_stats.json").write_text(json.dumps(all_stats, indent=1))
    (reports / "data_stats.md").write_text(to_markdown(all_stats, pool_sizes))
    print(f"wrote {reports/'data_stats.md'}")


if __name__ == "__main__":
    main()
