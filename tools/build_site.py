"""Build the public site (branch `site`) from web/index.html and the current results.

    .venv/bin/python tools/build_site.py OUT_DIR

Writes OUT_DIR/index.html (title + one-liner + gallery + viewer; everything else hidden) and OUT_DIR/data.json
(a snapshot of the results table with the coordinates needed to draw every picture).
"""
import json, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

# ---------- data snapshot
live = json.load(open(os.path.join(ROOT, "results", "live.json")))
try:
    live["experiments"] = json.load(open(os.path.join(ROOT, "results", "experiments.json")))
except OSError:
    live["experiments"] = None
table = {}
for k, e in live["table"].items():
    n = int(k)
    src = e.get("source", "")
    proved_trivial = n == 1 or 2 <= n <= 16 or round(n ** 0.25) ** 4 == n
    if src.startswith("grid") and not proved_trivial:      # hidden on the page anyway; keep the bound only
        table[k] = {"n": n, "s": e["s"], "source": src, "lower": e.get("lower"), "cubes": [{"c": q["c"], "R": q["R"]} for q in (e.get("cubes") or [])][:1] or None}
        continue
    keep = {x: e[x] for x in ("n", "s", "source", "lower", "found_at", "derived_from", "cited") if x in e}
    if e.get("cubes"):
        keep["cubes"] = [{"c": [round(v, 10) for v in q["c"]], "R": [[round(v, 10) for v in row] for row in q["R"]]} for q in e["cubes"]]
    if e.get("alt"):
        keep["alt"] = {"s": e["alt"]["s"], "source": e["alt"]["source"],
                       "cubes": [{"c": q["c"], "R": q["R"]} for q in (e["alt"].get("cubes") or [])]}
    table[k] = keep
snap = {"dim": 4, "ns": live["ns"], "updated": live["updated"], "started": live["started"], "uptime": 0, "workers": {},
        "steps_per_sec": 0, "events": [], "plan_taken": [], "minutes_per_n": 1, "table": table, "static": True,
        "snapshot_time": time.strftime("%Y-%m-%d %H:%M")}
json.dump(snap, open(os.path.join(out, "data.json"), "w"), separators=(",", ":"))

# ---------- page: title + one-liner + gallery; everything else hidden but present (the script expects it)
html = open(os.path.join(ROOT, "web", "index.html")).read()
start = html.index('  <p class="small" id="trivnote"')
start = html.index("\n", start) + 1
end = html.index("</main>")
html = html[:start] + '  <div style="display:none">\n' + html[start:end] + "  </div>\n" + html[end:]
intro = ('  <p class="byline" style="max-width:760px;margin:6px auto 8px">'
         'The following pictures show <i>n</i> unit tesseracts packed inside the smallest known tesseract (of side <i>s</i>). '
         'For all other values of <i>n</i>, the trivial packing is the best known.</p>\n'
         '  <p class="byline" style="max-width:760px;margin:0 auto 14px">'
         '4th dimension is visualized as time: two cubes can occupy the same space, just never at the same time.</p>\n'
         '  <p class="byline" style="margin:-8px auto 14px;font-size:15px;color:#555">Inspired by: <a href="https://erich-friedman.github.io/packing/cubincub/">cube packing</a> and <a href="https://kingbird.myphotos.cc/packing/squares_in_squares.html">square packing</a></p>\n')
html = html.replace("  <h1>Tesseracts in Tesseracts</h1>\n", "  <h1>Tesseracts in Tesseracts</h1>\n" + intro, 1)
html = html.replace("</main>",
                    '  <p class="byline" style="margin:24px auto 0">Click a picture for alternative visualizations.</p>\n</main>', 1)
html = html.replace("fetch('/results/live.json?t=' + Date.now()", "fetch('data.json?t=' + Date.now()")
html = html.replace("fetch('/plan.json?t=' + Date.now()", "fetch('data.json?t=' + Date.now()")
html = html.replace("poll(); setInterval(poll, 2000);",
                    "poll(); setInterval(poll, 300000);\n"
                    "")
# public captions: just "Found <date>." for anything found by search, experiment, or derived from a found packing
override = '''
const _caption = caption;
caption = function (e) {
  const r = _caption(e);
  if (e.found_at && !/^(grid|product)/.test(e.source || '')) {
    const d = new Date(e.found_at.replace(' ', 'T'));
    return ['Found ' + d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }) + '.', ''];
  }
  return r;
};
'''
html = html.replace("</script>\n</body>", override + "</script>\n</body>")
open(os.path.join(out, "index.html"), "w").write(html)
print("built", out, os.path.getsize(os.path.join(out, "data.json")) // 1024, "KB data")
