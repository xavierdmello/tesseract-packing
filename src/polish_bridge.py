"""Bridge to the augmented-Lagrangian + L-BFGS polisher from hockyy/tesseract-packing
(src/polish.cpp, built at /tmp/hockyy_tp/polish; E9 validated the pipeline).

polish(cubes, s) -> (s', cubes') or None on any failure. The caller always falls back
to the raw candidate, so a missing binary can never break the engine.
"""
import json, math, os, subprocess, tempfile

POLISH = os.environ.get("HPOLISH", "/tmp/hockyy_tp/polish")


def to_hockyy(cubes, s):
    n = len(cubes)
    C, R = [], []
    for q in cubes:
        C += [float(x) for x in q["c"]]
        R += [float(x) for row in q["R"] for x in row]
    U = []
    for i in range(n):
        ci = cubes[i]["c"]
        for j in range(i + 1, n):
            d = [cubes[j]["c"][k] - ci[k] for k in range(4)]
            nr = math.sqrt(sum(x * x for x in d)) or 1.0
            U += [x / nr for x in d]
    return {"d": 4, "n": n, "results": [{"s": s, "centers": C, "R": R, "u": U}]}


def from_hockyy(d):
    r = d["results"][0]
    n = d["n"]
    cubes = [{"c": r["centers"][4 * i:4 * i + 4],
              "R": [r["R"][16 * i + 4 * k:16 * i + 4 * k + 4] for k in range(4)]} for i in range(n)]
    return r["s"], cubes


def polish(cubes, s, timeout=90):
    if not os.path.exists(POLISH):
        return None
    fin = fout = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(to_hockyy(cubes, s), f, separators=(",", ":"))  # their parser greps "centers":[ with no space
            fin = f.name
        fout = fin + ".out"
        subprocess.run([POLISH, fin, fout, "1"], timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return from_hockyy(json.load(open(fout)))
    except Exception:
        return None
    finally:
        for p in (fin, fout):
            try:
                if p: os.unlink(p)
            except OSError:
                pass
