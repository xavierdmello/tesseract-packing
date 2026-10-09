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
  # refresh the README screenshot (needs a local server for the site folder)
  curl -s -o /dev/null http://127.0.0.1:8799/index.html || (cd ../4Dpacking-site && nohup python3 -m http.server 8799 >/dev/null 2>&1 &) ; sleep 2
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=1400,1150 \
     --force-device-scale-factor=1 --virtual-time-budget=6000 --screenshot=docs/screenshot.png http://127.0.0.1:8799/index.html >/dev/null 2>&1
  mkdir -p ../4Dpacking-site/docs && cp docs/screenshot.png ../4Dpacking-site/docs/screenshot.png
  git add docs/screenshot.png
  (cd ../4Dpacking-site && git add -A && { git diff --cached --quiet || { git commit -qm "Site snapshot $ts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "$ts site pushed"; }; })
  sleep $INTERVAL
done
