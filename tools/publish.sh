#!/bin/zsh
# Periodically publish: research branch (results, plan, notes) and the public site branch.
# Usage: tools/publish.sh [interval_seconds]   (runs forever; log in logs/publish.log)
cd "$(dirname "$0")/.."
INTERVAL=${1:-900}
while true; do
  ts=$(date "+%Y-%m-%d %H:%M")
  git add results/best plan.json notes README.md 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -qm "Results snapshot $ts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "$ts research pushed"
  fi
  .venv/bin/python tools/build_site.py ../4Dpacking-site >/dev/null
  (cd ../4Dpacking-site && git add -A && { git diff --cached --quiet || { git commit -qm "Site snapshot $ts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "$ts site pushed"; }; })
  sleep $INTERVAL
done
