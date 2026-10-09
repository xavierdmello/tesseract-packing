#!/bin/zsh
# Publisher. Every 60 s: if any record value changed, rebuild the public site and replace the single deploy
# commit on the `site` branch (force push; that branch is a deploy target, its history is not kept).
# Every 5 min: commit results/plan/notes to `research` if changed. Every 15 min: refresh the README screenshot.
cd "$(dirname "$0")/.."
SITE=../4Dpacking-site
last_research=0; last_shot=0; last_sig=""
while true; do
  now=$(date +%s); ts=$(date "+%Y-%m-%d %H:%M")
  sig=$(.venv/bin/python -c "import json;L=json.load(open('results/live.json'));print(','.join('%s:%.9f'%(k,v['s']) for k,v in sorted(L['table'].items(),key=lambda x:int(x[0]))))" 2>/dev/null)
  if [ -n "$sig" ] && [ "$sig" != "$last_sig" ]; then
    .venv/bin/python tools/build_site.py $SITE >/dev/null
    (cd $SITE && git add -A && { git diff --cached --quiet || { git commit -q --amend -m "Public site ($ts)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q -f && echo "$ts site updated"; }; })
    last_sig=$sig
  fi
  if (( now - last_shot >= 900 )); then
    curl -s -o /dev/null http://127.0.0.1:8799/index.html || (cd $SITE && nohup python3 -m http.server 8799 >/dev/null 2>&1 &); sleep 2
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=1400,1150 \
       --force-device-scale-factor=1 --virtual-time-budget=6000 --screenshot=docs/screenshot.png http://127.0.0.1:8799/index.html >/dev/null 2>&1
    mkdir -p $SITE/docs && cp docs/screenshot.png $SITE/docs/screenshot.png
    last_shot=$now; last_sig="force-site-rebuild-next-round"
  fi
  if (( now - last_research >= 300 )); then
    git add results/best results/experiments.json plan.json notes README.md docs/screenshot.png web/index.html tools/build_site.py 2>/dev/null
    if ! git diff --cached --quiet; then
      git commit -qm "Results snapshot $ts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "$ts research pushed"
    fi
    last_research=$now
  fi
  sleep 60
done
