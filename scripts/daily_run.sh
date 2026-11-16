#!/usr/bin/env bash
# One day's worth of experiment work on the free tier. Resumable: run it once a day
# (in a fresh session: `scripts/sync_runs.sh pull` first) until everything is done.
#   scripts/daily_run.sh baseline
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-baseline}
for split in dev test; do
  for ds in musique hotpotqa qasper ambigqa; do
    .venv/bin/python scripts/run_agent.py --dataset "$ds" --split "$split" --run "$RUN"
    if [ $? -eq 3 ]; then echo "== quota reached; syncing and stopping for today"; scripts/sync_runs.sh push; exit 0; fi
  done
done
scripts/sync_runs.sh push
echo "== all splits complete for run '$RUN'"
