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
one_liner = ('  <p class="byline" style="max-width:760px;margin:6px auto 14px">Each picture is a 3D movie of a 4D packing: '
             'the 4th dimension plays as time, so every tesseract is a cube that appears, exists for a while and vanishes; '
             'the bars underneath show when each one exists. Click a picture to scrub through time or rotate it. '
             '<span id="snap" style="color:#777"></span> · <a href="https://github.com/xavierdmello/tesseract-packing">code &amp; data</a>'
             ' · related: <a href="https://github.com/hockyy/tesseract-packing">hockyy/tesseract-packing</a> (better values for n = 26–30)</p>\n')
html = html.replace("  <h1>Tesseracts in Tesseracts</h1>\n", "  <h1>Tesseracts in Tesseracts</h1>\n" + one_liner, 1)
html = html.replace("fetch('/results/live.json?t=' + Date.now()", "fetch('data.json?t=' + Date.now()")
html = html.replace("fetch('/plan.json?t=' + Date.now()", "fetch('data.json?t=' + Date.now()")
html = html.replace("poll(); setInterval(poll, 2000);",
                    "poll(); setInterval(poll, 300000);\n"
                    "setTimeout(() => { if (LIVE && LIVE.snapshot_time) document.getElementById('snap').textContent = 'Results as of ' + LIVE.snapshot_time + '.'; }, 1500);")
open(os.path.join(out, "index.html"), "w").write(html)
print("built", out, os.path.getsize(os.path.join(out, "data.json")) // 1024, "KB data")
