#!/usr/bin/env bash
# Back up experiment outputs (runs/ + compressed LLM cache) to the `runs-data` branch,
# so progress survives the cloud container being recycled.
#   scripts/sync_runs.sh push      # save current runs + cache
#   scripts/sync_runs.sh pull      # restore them into a fresh checkout
set -euo pipefail
cd "$(dirname "$0")/.."
BR=runs-data; WT=.sync
git fetch -q origin "$BR" 2>/dev/null || true
if [ ! -d "$WT" ]; then
  if git rev-parse -q --verify "origin/$BR" >/dev/null; then
    git worktree add -q -B "$BR" "$WT" "origin/$BR"
  else
    git worktree add -q --detach "$WT" && (cd "$WT" && git checkout -q --orphan "$BR" && git rm -rqf . || true)
  fi
fi
case "${1:-push}" in
  push)
    mkdir -p "$WT/runs" "$WT/cache"
    rm -rf "$WT/runs" && mkdir -p "$WT/runs" && { [ -d runs ] && cp -r runs/. "$WT/runs/" || true; }
    if [ -f .cache/llm.sqlite ]; then
      python3 -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close()" \
        .cache/llm.sqlite "$WT/cache/llm.sqlite" && gzip -nf "$WT/cache/llm.sqlite"
    fi
    cd "$WT" && git add -A && { git diff --cached --quiet && echo "nothing new"; } || {
      git commit -qm "runs sync $(date -u +%Y-%m-%dT%H:%MZ)" && git push -q -u origin "$BR" && echo "pushed $BR"; } ;;
  pull)
    mkdir -p runs .cache
    cp -r "$WT/runs/." runs/
    [ -f "$WT/cache/llm.sqlite.gz" ] && gunzip -c "$WT/cache/llm.sqlite.gz" > .cache/llm.sqlite
    echo "restored runs/ and .cache/llm.sqlite" ;;
esac
