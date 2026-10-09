"""Sequential multi-core search: all workers attack ONE n at a time, then move on.

    python src/engine.py --ns 17-100 --minutes 5 --cpu-workers 11 --gpu-workers 1

Collaboration: workers share a "best so far" board (Manager dict).  Every
improvement goes to the main process, is certified in float64 (LP separating-
hyperplane test per pair + container check), stored, and published back.  On
every restart a worker picks a starting configuration from a mix of strategies:
  best   – jitter the current best packing for n (one cube re-placed at random)
  grow   – best packing for n-1 plus one random cube
  shrink – best packing for n+1 minus one random cube (if known)
  random – fresh random configuration
Each worker has a different strategy mix and learning rate, so the team covers
both exploitation and exploration.

Observability: terminal dashboard (stdout), results/live.json (web UI at
http://localhost:PORT), results/best/d4_n{N}.json, results/events.log.
"""
import argparse, json, math, os, queue, random, sys, threading, time, traceback
import multiprocessing as mp
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

STRATS = ["best", "grow", "shrink", "random", "morph", "seed"]
MIXES = [  # per-worker strategy probabilities (best, grow, shrink, random, morph)
    # morph lost the 3D n=12 A/B test (2.962 vs 2.935 in 3 min), so it gets a small exploration share
    # seed = structured starts from src/theory_seeds.py (only for n where a generator exists)
    (0.6, 0.2, 0.1, 0.1, 0.0, 0.1), (0.3, 0.3, 0.1, 0.2, 0.1, 0.2), (0.1, 0.1, 0.0, 0.5, 0.3, 0.1), (0.4, 0.4, 0.2, 0.0, 0.0, 0.2),
    (0.2, 0.3, 0.2, 0.2, 0.1, 0.3), (0.7, 0.1, 0.1, 0.1, 0.0, 0.1)]


# =============================================================== worker
def worker_main(wid, D, task_q, out_q, board, device, batch, allowed=None, morph_cfg=None, native=False):
    import torch
    import numpy as np
    from kernel import Batch
    torch.set_num_threads(1)
    dtype = torch.float32 if device == "mps" else torch.float64
    rng = np.random.default_rng(wid * 7919 + int(time.time()))
    mix = np.array(MIXES[wid % len(MIXES)], float)
    if allowed and set(allowed) != set(STRATS):
        mix = np.array([st in allowed for st in STRATS], float)   # explicit strategy list: equal weights
    lr_choices = [0.003, 0.006, 0.012, 0.02]
    cfg_cache = {}
    rot_cache = {}

    def seed_gen(n):
        """Return f(s_start) -> (s, cubes) for structured seeds, or None if there is no generator for n."""
        if D != 4: return None
        import theory_seeds as TS
        if 18 <= n <= 24: return lambda s0: TS.hybrid17_seed(n, s0, rng=rng)
        if 26 <= n <= 41: return lambda s0: TS.cross_cell_seed(n, s0, rng=rng)
        if 101 <= n <= 120:
            path = os.path.join(ROOT, "results_aux2", "best", "d2_n10.json")
            if not os.path.exists(path): return None
            e2 = json.load(open(path))
            sq = [((q["c"][0], q["c"][1]), math.atan2(q["R"][1][0], q["R"][0][0])) for q in e2["cubes"]]
            return lambda s0: TS.product_frame_seed(sq, sq, n - 100, max(s0, e2["s"]), rng=rng)
        return None

    def p_of(m, R):
        key = tuple(round(x, 9) for row in R for x in row)
        if key not in rot_cache:
            from kernel import fit_p
            rot_cache[key] = fit_p([R])[0][0]
        return rot_cache[key]

    def to_tensor_cfg(m, cfg):
        """config dict -> (c, p) tensors (uses stored quaternion parameters 'p')."""
        key = (len(cfg["cubes"]), cfg["s"])
        if key not in cfg_cache:
            if len(cfg_cache) > 16: cfg_cache.clear()
            c = torch.tensor([q["c"] for q in cfg["cubes"]], **m.kw)
            p = torch.tensor([q["p"] for q in cfg["cubes"]], **m.kw)
            cfg_cache[key] = (c, p)
        return cfg_cache[key]

    def run(task):
        n, deadline = task["n"], task["deadline"]
        B = batch
        m = Batch(D, n, B, device, dtype)
        kw = m.kw
        lr = float(rng.choice(lr_choices))
        s0 = torch.zeros(B, **kw)
        fac = torch.full((B,), 0.996, **kw)
        dr = torch.zeros(B, **kw)              # per-round decrease of the rounding radius (morph strategy)
        mph = torch.zeros(B, dtype=torch.long, device=m.dev)   # morph phase: 0 rigid, 1 compress balls, 2 morph
        jamc = torch.zeros(B, dtype=torch.long, device=m.dev)
        elem_best = torch.full((B,), float("inf"), **kw)
        hit_s = task.get("hit_s")
        ppid = os.getppid()
        stalls = torch.zeros(B, dtype=torch.long, device=m.dev)
        strat_of = np.zeros(B, int)
        stats = {"restarts": 0, "feasible": 0, "by_strat": {s: 0 for s in STRATS}, "wins": {s: 0 for s in STRATS},
                 "hits": {s: 0 for s in STRATS}, "ends": {s: 0 for s in STRATS}}

        def board_get(k):
            v = board.get((D, k)); return v if v and v.get("cubes") else None

        def init(idx):
            k = len(idx)
            if k == 0: return
            best_n, best_m1, best_p1 = board_get(n), board_get(n - 1), board_get(n + 1)
            cur = (board.get((D, n)) or {}).get("s", n ** 0.25 + 1)
            sg = seed_gen(n)
            avail = np.array([best_n is not None, best_m1 is not None, best_p1 is not None, True, True, sg is not None], float)
            pr = mix * avail
            if pr.sum() == 0: pr = avail
            pr = pr / pr.sum()
            choice = torch.tensor(rng.choice(len(STRATS), size=k, p=pr), device=m.dev)
            for si, st in enumerate(STRATS):
                E = idx[choice == si]
                kk = len(E)
                if kk == 0: continue
                strat_of[E.cpu().numpy()] = si
                stats["by_strat"][st] += kk
                lo, hi = {"random": (0.04, 0.25), "morph": (0.15, 0.45)}.get(st, (0.003, 0.04))
                s_start = cur * (1 + lo + (hi - lo) * torch.rand(kk, **kw))
                s0[E] = s_start; m.s[E] = s_start
                cr = torch.rand(kk, n, D, **kw) * (s_start[:, None, None] - 1) + 0.5
                prr = m.random_p(kk, n)
                ar = torch.arange(kk, device=m.dev)
                m.r[E] = 0.5 if st == "morph" else 0.0
                mph[E] = 1 if st == "morph" else 0; jamc[E] = 0
                mk = (morph_cfg or {}).get("k", (8, 40))
                dr[E] = (0.5 / torch.randint(mk[0], mk[1], (kk,), device=m.dev).to(m.dt)) if st == "morph" else 0.0
                if st == "seed":
                    cs_, ps_ = [], []
                    for t_ in range(kk):
                        s_seed, cubes_ = sg(float(s_start[t_]))
                        s_start[t_] = max(float(s_start[t_]), s_seed)
                        cs_.append(torch.tensor([q["c"] for q in cubes_], **kw))
                        ps_.append(torch.tensor([p_of(m, q["R"]) for q in cubes_], **kw))
                    s0[E] = s_start; m.s[E] = s_start
                    c, p = torch.stack(cs_), torch.stack(ps_)
                elif st in ("random", "morph"):
                    c, p = cr, prr
                else:
                    src = {"best": best_n, "grow": best_m1, "shrink": best_p1}[st]
                    sc, sp = to_tensor_cfg(m, src)
                    scale = (s_start / src["s"])[:, None, None]
                    sc = (sc[None] - src["s"] / 2) * scale + s_start[:, None, None] / 2
                    sp = sp[None].expand(kk, -1, -1)
                    if st == "grow":       # add one random cube
                        c = torch.cat([sc, cr[:, :1]], 1); p = torch.cat([sp, prr[:, :1]], 1)
                    elif st == "shrink":   # drop one random cube
                        drop = torch.randint(0, n + 1, (kk, 1), device=m.dev)
                        keep = torch.arange(n, device=m.dev)[None] + (torch.arange(n, device=m.dev)[None] >= drop).long()
                        c = torch.gather(sc, 1, keep[..., None].expand(-1, -1, D))
                        p = torch.gather(sp, 1, keep[..., None].expand(-1, -1, m.NP))
                    else:                  # jitter + re-place one random cube
                        c, p = sc.clone(), sp.clone()
                        j = torch.randint(0, n, (kk,), device=m.dev)
                        c[ar, j] = cr[ar, j]; p[ar, j] = prr[ar, j]
                    amp = torch.tensor([0.0, 0.01, 0.03, 0.08], **kw)[torch.randint(0, 4, (kk,), device=m.dev)][:, None, None]
                    c = c + amp * torch.randn_like(c); p = p + 2 * amp * torch.randn_like(p)
                m.c[E] = c; m.p[E] = p
            m.reset_planes(idx)

        with torch.no_grad():
            init(torch.arange(B, device=m.dev))
        use_native = native and D == 4 and device == "cpu"
        if use_native:
            import nkernel
            na = nkernel.NativeAdam(m, lr, prune=True)
        else:
            for t in m.params(): t.requires_grad_(True)
            opt = torch.optim.Adam(m.params(), lr=lr)
        target = (board.get((D, n)) or {}).get("s", float("inf"))
        t0 = time.time(); last_rep = 0; last_board = 0; steps = 0
        task_best = float("inf")
        inner = 100
        while time.time() < deadline:
            if os.getppid() != ppid: os._exit(0)          # parent died: never linger as an orphan
            if use_native:
                pen, mx = na.steps(inner)
            else:
                for _ in range(inner):
                    opt.zero_grad(set_to_none=True)
                    pen, _ = m.violation(); pen.sum().backward(); opt.step()
            steps += inner
            with torch.no_grad():
                if not use_native: pen, mx = m.violation()
                feas = mx < m.margin * 0.5
                rigid = m.r <= 0
                stalls += (~feas & rigid).long()
                # morph phase 1: compress balls fast until jammed twice -> phase 2
                c1 = (mph == 1)
                jamc += (c1 & ~feas).long()
                go = (c1 & (jamc >= 2)).nonzero().flatten()
                mph[go] = 2
                fast = (c1 & feas).nonzero().flatten()
                if len(fast):
                    s_new = 1 + (m.s[fast] - 1) * 0.975
                    m.c[fast] = m.c[fast] * (s_new / m.s[fast])[:, None, None]; m.s[fast] = s_new
                # morph phase 2: walls push while feasible (below), give way slightly when jammed; radius shrinks
                jam = (~feas & (mph == 2)).nonzero().flatten()
                if len(jam):
                    m.c[jam] = m.c[jam] * 1.002; m.s[jam] = m.s[jam] * 1.002
                c2 = mph == 2
                noise = (morph_cfg or {}).get("noise", 0.0)
                if noise > 0 and c2.any():             # annealing: jiggle that fades as the bodies harden
                    i2 = c2.nonzero().flatten()
                    amp = (noise * m.r[i2] / 0.5)[:, None, None]
                    m.c[i2] += amp * torch.randn(len(i2), n, D, **kw)
                    m.p[i2] += 2 * amp * torch.randn(len(i2), n, m.NP, **kw)
                m.r[c2] = (m.r[c2] - dr[c2]).clamp(min=0)
                done = (c2 & (m.r <= 0)).nonzero().flatten()
                mph[done] = 0; dr[done] = 0
                fidx = (feas & (mph != 1)).nonzero().flatten()
                ridx = (feas & rigid).nonzero().flatten()
                if len(fidx):
                    stats["feasible"] += len(ridx)
                if len(ridx):
                    elem_best[ridx] = torch.minimum(elem_best[ridx], m.s[ridx])
                    k = int(ridx[m.s[ridx].argmin()])
                    sk = float(m.s[k])
                    task_best = min(task_best, sk)
                    if sk < target - 1e-9:
                        out_q.put(("found", wid, D, n, m.config(k), STRATS[strat_of[k]]))
                        stats["wins"][STRATS[strat_of[k]]] += 1
                        target = sk
                if len(fidx):
                    s_new = 1 + (m.s[fidx] - 1) * fac[fidx]
                    m.c[fidx] = m.c[fidx] * (s_new / m.s[fidx])[:, None, None]
                    m.s[fidx] = s_new
                    stalls[fidx] = 0
                bidx = ((stalls > 0) & (stalls % 4 == 0)).nonzero().flatten()
                if len(bidx):   # back off half a step, refine step, shake
                    s_back = 1 + (m.s[bidx] - 1) / fac[bidx].sqrt()
                    m.c[bidx] = m.c[bidx] * (s_back / m.s[bidx])[:, None, None]
                    m.s[bidx] = s_back
                    fac[bidx] = fac[bidx].sqrt().clamp(max=1 - 1e-7)
                    m.c[bidx] += 0.005 * torch.randn(len(bidx), n, D, **kw)
                    m.p[bidx] += 0.01 * torch.randn(len(bidx), n, m.NP, **kw)
                dead = (stalls >= 16).nonzero().flatten()
                if len(dead):
                    stats["restarts"] += len(dead)
                    for e_ in dead.tolist():           # outcome of each finished start, per strategy
                        st_ = STRATS[strat_of[e_]]; stats["ends"][st_] += 1
                        if hit_s is not None and float(elem_best[e_]) <= hit_s: stats["hits"][st_] += 1
                    elem_best[dead] = float("inf")
                    init(dead); stalls[dead] = 0; fac[dead] = 0.996
                    if use_native: na.reset(dead)
                    else:
                        for t in m.params():
                            st = opt.state.get(t)
                            if st: st["exp_avg"][dead] = 0; st["exp_avg_sq"][dead] = 0
            if time.time() - last_board > 2:
                last_board = time.time()
                target = min(target, (board.get((D, n)) or {}).get("s", float("inf")))
                if board.get(("stop", task.get("tid"))): break      # main ended this task early
            if time.time() - last_rep > 2:
                last_rep = time.time()
                out_q.put(("progress", wid, {"n": n, "device": device, "batch": B, "lr": lr,
                    "task_best": task_best if task_best < 1e9 else None, "median_s": float(m.s.median()),
                    "target": target, "restarts": stats["restarts"], "feasible": stats["feasible"],
                    "by_strat": stats["by_strat"], "wins": stats["wins"], "mix": list(mix),
                    "hits": stats["hits"], "ends": stats["ends"],
                    "steps_per_sec": steps * B / max(time.time() - t0, 1e-9)}))

    out_q.put(("hello", wid, device))
    while True:
        task = task_q.get()
        if task is None: break
        out_q.put(("start", wid, {"n": task["n"], "t0": time.time(), "deadline": task["deadline"], "minutes": task.get("minutes", 1), "tid": task.get("tid")}))
        try:
            run(task)
        except Exception:
            out_q.put(("error", wid, traceback.format_exc()))
            time.sleep(10)                     # never spin on a crashing task
        out_q.put(("idle", wid, task["n"]))


# =============================================================== main
# Manual controls (web UI → HTTP handler → main loop). Persisted in results/control.json.
CONTROL = {"forced": [], "queue": [], "blocked": [], "stop": [], "lock": threading.Lock()}


def control_save():
    try:
        with open(os.path.join(ROOT, "results", "control.json.tmp"), "w") as f:
            json.dump({k: CONTROL[k] for k in ("queue", "blocked")}, f)
        os.replace(os.path.join(ROOT, "results", "control.json.tmp"), os.path.join(ROOT, "results", "control.json"))
    except Exception:
        pass


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
    def end_headers(self):
        self.send_header("Cache-Control", "no-store"); super().end_headers()
    def do_POST(self):
        if self.path == "/control":
            try:
                ln = int(self.headers.get("Content-Length", 0) or 0)
                b = json.loads(self.rfile.read(ln) or b"{}")
                a = b.get("action"); n = int(b["n"]) if "n" in b else None
                mins = max(0.5, min(60.0, float(b.get("minutes", 10))))
                if n is not None and n < 1:
                    self.send_error(400, "n must be a positive integer"); return
                with CONTROL["lock"]:
                    if a == "force": CONTROL["forced"].append({"n": n, "minutes": mins})
                    elif a == "queue_add": CONTROL["queue"].append({"n": n, "minutes": mins})
                    elif a == "queue_remove": CONTROL["queue"] = [e for i, e in enumerate(CONTROL["queue"]) if i != int(b.get("index", -1))]
                    elif a == "block": CONTROL["blocked"] = sorted(set(CONTROL["blocked"]) | {n})
                    elif a == "unblock": CONTROL["blocked"] = [x for x in CONTROL["blocked"] if x != n]
                    elif a == "stop": CONTROL["stop"].append(int(b["wid"]))
                    else: raise ValueError(a)
                    control_save()
                    out = json.dumps({k: CONTROL[k] for k in ("forced", "queue", "blocked")}).encode()
            except Exception as ex:
                self.send_error(400, str(ex)); return
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out); return
        if self.path != "/auto":
            self.send_error(404); return
        import subprocess as sp
        try:
            ln = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            self.send_error(400); return
        # The planner process always runs (it is also the production watchdog). The button only flips plan.json's
        # "auto" flag: ON = the planner writes plan.json; OFF = a researcher owns plan.json (planner just guards).
        running = sp.run(["pgrep", "-f", "tools/autoplan.py"], capture_output=True, text=True).stdout.split()
        if not running:
            sp.Popen([".venv/bin/python", "tools/autoplan.py"], stdout=open("logs/autoplan.log", "a"),
                     stderr=open("logs/autoplan.log", "a"), start_new_session=True)
        try:                                     # reflect the flag / minutes in plan.json (bumps version -> reload)
            p = json.load(open("plan.json"))
            if "auto" in body: p["auto"] = bool(body["auto"])
            if "minutes" in body: p["auto_minutes"] = max(0.5, min(60.0, float(body["minutes"])))
            if "research_cores" in body: p["research_cores"] = int(max(0, min(12, int(body["research_cores"]))))
            p["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            json.dump(p, open("plan.json.tmp", "w"), indent=2, ensure_ascii=False)
            os.replace("plan.json.tmp", "plan.json")
            want = {"auto": p.get("auto", False), "minutes": p.get("auto_minutes", 10), "research_cores": p.get("research_cores", 4)}
        except Exception:
            want = {}
        out = json.dumps(want).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def parse_ns(spec):
    out = []
    for part in spec.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f)
    os.replace(tmp, path)


def certify_inflate(G, cubes, s):
    """Certify; if a tiny violation remains (float32 / margin), spread centres by lambda and re-measure s."""
    import numpy as np
    ok, cont, sep = G.certify(cubes, s, tol=1e-10)
    if ok: return True, cubes, s, sep
    for lam in (1 + 1e-7, 1 + 1e-6, 1 + 1e-5, 1 + 1e-4):
        cs = [np.asarray(q["c"]) for q in cubes]
        mid = np.full(len(cs[0]), s / 2)
        new = [{**q, "c": (mid + lam * (c - mid)).tolist()} for c, q in zip(cs, cubes)]
        V = np.vstack([G.cube_vertices(q["c"], q["R"]) for q in new])
        lo, hi = V.min(0), V.max(0)
        s2 = float((hi - lo).max())
        new = [{**q, "c": (np.asarray(q["c"]) - lo).tolist()} for q in new]
        ok, cont, sep = G.certify(new, s2, tol=1e-10)
        if ok: return True, new, s2, sep
    return False, cubes, s, sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="17-100")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--cpu-workers", type=int, default=11)
    ap.add_argument("--gpu-workers", type=int, default=1)
    ap.add_argument("--cpu-batch", type=int, default=32)
    ap.add_argument("--gpu-batch", type=int, default=1024)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--stall-seconds", type=float, default=20, help="end an n early after this long without a record")
    ap.add_argument("--morph-k", default="8,40", help="morph lasts K rounds of 100 steps, K uniform in this range")
    ap.add_argument("--morph-noise", type=float, default=0.0, help="annealing noise amplitude during morph")
    ap.add_argument("--native", action="store_true", help="use the C kernel (D=4, CPU)")
    ap.add_argument("--hit-s", type=float, default=None, help="experiments: count starts that reach s <= this")
    ap.add_argument("--label", default="", help="one-line description of this run (shown on the website for experiments)")
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--strategies", default=",".join(STRATS))
    ap.add_argument("--results", default="results", help="results directory (relative to repo)")
    args = ap.parse_args()
    import geometry as G
    D = args.dim
    RES = os.path.join(ROOT, args.results); args.res = RES
    os.makedirs(os.path.join(RES, "best"), exist_ok=True)
    ns = parse_ns(args.ns)
    NMAX = max(max(ns) + 1, 81)

    # ---------- best-known table: constructions, overridden by saved results
    if D == 4:
        table = G.known_constructions_4d(NMAX, root=ROOT)
    else:
        table = {}
        for n in range(1, NMAX + 1):
            sg, cubes = G.grid_packing(n, D)
            table[n] = {"n": n, "s": sg, "cubes": cubes, "source": "grid (trivial)", "certified": True}
    for n in table:
        path = os.path.join(RES, "best", f"d{D}_n{n}.json")
        if os.path.exists(path):
            saved = json.load(open(path))
            if saved.get("construction"): continue          # constructions are rebuilt from scratch each start
            if saved["s"] < table[n]["s"]: table[n] = saved
            elif table[n].get("cited") and saved.get("cubes") and (not table[n].get("alt") or saved["s"] < table[n]["alt"]["s"]):
                table[n]["alt"] = {k: saved[k] for k in ("s", "cubes", "source")}
        table[n].setdefault("history", [])
        table[n]["lower"] = max(n ** (1 / D), 2.0 if n >= 2 else 1.0)
        table[n].setdefault("search", None)

    from kernel import fit_p
    p_memo = {}                                  # rotation -> quaternion-pair parameters (fitted once per distinct rotation)
    for n in table:
        cubes = table[n]["cubes"]
        if cubes and D != 4:
            for q in cubes:
                if "p" not in q:
                    from scipy.linalg import logm
                    import numpy as np
                    A = np.real(logm(np.asarray(q["R"]))); q["p"] = A[np.triu_indices(D, 1)].tolist()
        elif cubes and any("p" not in q for q in cubes):
            for q in cubes:
                if "p" in q: continue
                key = tuple(round(x, 9) for row in q["R"] for x in row)
                if key not in p_memo:
                    ps, err = fit_p([q["R"]])
                    p_memo[key] = ps[0]
                    if err > 1e-12: print(f"warning: rotation fit error {err:.1e} for n={n}", file=sys.stderr)
                q["p"] = p_memo[key]

    mgr = mp.Manager()
    board = mgr.dict()
    for n in table: board[(D, n)] = {"s": table[n]["s"], "cubes": table[n]["cubes"]}

    task_q, out_q = mp.Queue(), mp.Queue()
    procs, workers = [], {}
    specs = [("cpu", args.cpu_batch)] * args.cpu_workers + [("mps", args.gpu_batch)] * args.gpu_workers
    for wid, (dev, b) in enumerate(specs):
        pr = mp.Process(target=worker_main, args=(wid, D, task_q, out_q, board, dev, b, args.strategies.split(","),
                                                      {"k": tuple(int(x) for x in args.morph_k.split(",")), "noise": args.morph_noise}, args.native), daemon=True)
        pr.start(); procs.append(pr); workers[wid] = {"state": "starting", "device": dev}
    NW = len(specs)

    os.chdir(ROOT)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Quiet)
    httpd.valid_ns = set(table)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    events = []
    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        events.append(line); del events[:-300]
        with open(os.path.join(RES, "events.log"), "a") as f: f.write(line + "\n")

    import signal
    def _term(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    t_start = time.time()
    log(f"engine start (farm mode): {NW} workers, each on its own n, {args.minutes} min per task, n in {ns[0]}..{ns[-1]}")
    import glob
    searches = {n: 0 for n in ns}          # completed tasks per n
    improved_last = {}                     # n -> did its last task improve it
    streak = {}                            # consecutive improving tasks
    hot_until = {}                         # n -> time until which n is "hot" (a neighbour just set a record)
    active = {}                            # wid -> {"n", "t0", "deadline", "s_before"}
    last_write = 0; last_logged = {}; last_import = 0

    # ---- the search order comes ONLY from plan.json (written by the researcher); it is re-read on every assignment
    plan_state = {"version": None, "taken": []}

    def read_plan():
        if args.results != "results":              # experiments never follow the production plan
            return {"updated": "fixed", "queue": [{"n": n} for n in ns]}
        try:
            p = json.load(open(os.path.join(ROOT, "plan.json")))
        except Exception as ex:
            log(f"plan.json unreadable ({ex}); keeping previous plan"); return None
        if p.get("updated") != plan_state["version"]:
            plan_state["version"] = p.get("updated"); plan_state["taken"] = []
            log(f"new research plan loaded ({p.get('updated')}): {[q['n'] for q in p.get('queue', [])]}")
        return p

    MAX_LONG = 2                                   # at most this many cores on long (> 1 min) tasks

    mq_state = {"i": 0}

    def ensure_n(n):
        """Unbounded n: create a table entry on demand (trivial grid packing) so any n can be searched."""
        if n in table: return
        sg, cubes = G.grid_packing(n, D)
        for q in cubes: q["p"] = [1.0, 0, 0, 0, 1.0, 0, 0, 0]
        table[n] = {"n": n, "s": sg, "cubes": cubes, "source": "grid (trivial)", "certified": True, "history": [],
                    "lower": max(n ** (1 / D), 2.0 if n >= 2 else 1.0), "search": None}
        board[(D, n)] = {"s": sg, "cubes": cubes}
        searches.setdefault(n, 0)
        log(f"manual: added n={n} (starts from the trivial grid, s={sg:g})")

    def next_task():
        """(n, minutes): forced runs first, then the manual queue, then the plan. Blocked n are never scheduled."""
        busy = {a["n"] for a in active.values()}
        with CONTROL["lock"]:
            blocked = set(CONTROL["blocked"])
            for e in CONTROL["forced"] + CONTROL["queue"]: ensure_n(e["n"])
            while CONTROL["forced"]:
                e = CONTROL["forced"].pop(0)
                if e["n"] in table and e["n"] not in busy:
                    log(f"manual: forced run n={e['n']} for {e['minutes']} min"); return e["n"], e["minutes"]
            mq = [e for e in CONTROL["queue"] if e["n"] in table and e["n"] not in busy and e["n"] not in blocked]
            if mq:
                e = mq[mq_state["i"] % len(mq)]; mq_state["i"] += 1
                return e["n"], e["minutes"]
        p = read_plan() or {"queue": []}
        busy = busy | blocked
        n_long = sum(1 for a in active.values() if a.get("minutes", args.minutes) > args.minutes)
        max_long = int(p.get("max_long", MAX_LONG))
        q = [e for e in p.get("queue", []) if e["n"] in table]
        mins_of = lambda e: min(60.0, float(e.get("minutes", args.minutes)))
        ok = lambda e: e["n"] not in busy and (mins_of(e) <= args.minutes or n_long < max_long)
        for e in q:
            if e["n"] not in plan_state["taken"] and ok(e):
                plan_state["taken"].append(e["n"]); return e["n"], mins_of(e)
        for e in q:                                    # plan exhausted: cycle until the researcher updates it
            if ok(e):
                plan_state["taken"] = [e["n"]]; return e["n"], mins_of(e)
        for e in q:                                    # long slots full: run a free plan entry as a short task
            if e["n"] not in busy:
                return e["n"], args.minutes
        free = [n for n in ns if n not in busy]        # never put two workers on the same n (or a blocked one)
        return (free[0] if free else ns[0]), args.minutes

    def assign():
        n, mins = next_task()
        task_q.put({"n": n, "deadline": time.time() + mins * 60, "tid": f"{n}-{time.time()}", "hit_s": args.hit_s, "minutes": mins})
        return n

    def accept(m_n, cfg_cubes, cfg_s, how, wid=None):
        if m_n not in table:                         # outside this engine's range (e.g. an experiment on other n)
            return False
        if cfg_s >= table[m_n]["s"] - 1e-9: return False
        # hockyy pipeline promotion (E9): squeeze the raw candidate with their augmented-Lagrangian
        # + L-BFGS polisher first; keep it only if strictly better and certifiable.
        try:
            from polish_bridge import polish
            pr = polish(cfg_cubes, cfg_s)
            if pr and pr[0] < cfg_s - 1e-12:
                ok2, c2, s2, sep2 = certify_inflate(G, pr[1], pr[0])
                if ok2 and s2 < table[m_n]["s"] - 1e-9:
                    ok, cubes, s_cert, sep = True, c2, s2, sep2
                    how += " + AL polish (hockyy)"
                else:
                    ok = False
            else:
                ok = False
        except Exception:
            ok = False
        if not ok:
            ok, cubes, s_cert, sep = certify_inflate(G, cfg_cubes, cfg_s)
        if not ok:
            log(f"n={m_n}: candidate s={cfg_s:.6f} failed certification (sep {sep:.1e}) – discarded"); return False
        if s_cert >= table[m_n]["s"] - 1e-9: return False
        if D == 4 and any("p" not in q for q in cubes):    # imported packings: fit the quaternion parameters workers need
            ps, err = fit_p([q["R"] for q in cubes])
            cubes = [{**q, "p": pp} for q, pp in zip(cubes, ps)]
        old = table[m_n]["s"]
        table[m_n] = {"n": m_n, "s": s_cert, "cubes": cubes, "certified": True, "source": how,
                      "found_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cert": {"min_pair_separation": sep},
                      "lower": table[m_n]["lower"], "history": table[m_n].get("history", []) + [[time.time(), s_cert]],
                      "search": table[m_n].get("search"), "previous_s": old if "previous_s" not in table[m_n] else table[m_n]["previous_s"]}
        board[(D, m_n)] = {"s": s_cert, "cubes": cubes}
        atomic_json(os.path.join(RES, "best", f"d{D}_n{m_n}.json"), table[m_n])
        hot_until[m_n + 1] = time.time() + 600
        for k in range(m_n - 1, 0, -1):    # s(k) <= s(m_n) for k < m_n: drop cubes
            if k in table and table[k]["s"] > s_cert + 1e-12:
                table[k] = {**table[k], "s": s_cert, "cubes": cubes[:k], "certified": True,
                            "source": f"n={m_n} packing with {m_n - k} cube(s) removed",
                            "found_at": time.strftime("%Y-%m-%d %H:%M:%S"), "derived_from": m_n}
                board[(D, k)] = {"s": s_cert, "cubes": cubes[:k]}
                atomic_json(os.path.join(RES, "best", f"d{D}_n{k}.json"), table[k])
        lt, ls = last_logged.get(m_n, (0, old))
        if ls - s_cert > 1e-3 or time.time() - lt > 20:
            log(f"n={m_n}: {ls:.6f} -> {s_cert:.6f}  ({how})")
            last_logged[m_n] = (time.time(), s_cert)
        return True

    args.plan_state = plan_state
    for _ in range(NW): assign()
    try:                                              # restore the manual queue / blocked list
        c0 = json.load(open(os.path.join(RES, "control.json")))
        CONTROL["queue"], CONTROL["blocked"] = c0.get("queue", []), c0.get("blocked", [])
    except Exception:
        pass
    args.control = CONTROL
    try:
        while True:
            # manual controls: stop requested runs; a forced run preempts the most recently started run
            with CONTROL["lock"]:
                stops = CONTROL["stop"][:]; CONTROL["stop"].clear()
                if CONTROL["forced"] and not stops and active:
                    victim = max(active, key=lambda w: active[w].get("t0", 0))
                    if not active[victim].get("preempted"):
                        stops = [victim]; active[victim]["preempted"] = True
            for w in stops:
                if w in active and active[w].get("tid"):
                    board[("stop", active[w]["tid"])] = True
                    log(f"manual: stopped worker {w} (n={active[w]['n']})")
            try: msg = out_q.get(timeout=0.3)
            except queue.Empty: msg = None
            if msg:
                kind = msg[0]
                if kind == "hello": workers[msg[1]] = {"state": "ready", "device": msg[2]}
                elif kind == "start":
                    wid, info = msg[1], msg[2]
                    active[wid] = {**info, "s_before": table[info["n"]]["s"]}
                    workers[wid] = {"state": "running", "device": workers[wid].get("device"), "n": info["n"], "t0": info["t0"],
                                    "deadline": info["deadline"], "minutes": info.get("minutes", 1), "task_best": None, "target": table[info["n"]]["s"]}
                elif kind == "progress":
                    workers[msg[1]] = {**workers.get(msg[1], {}), "state": "running", **msg[2]}
                elif kind == "idle":
                    wid = msg[1]; a = active.pop(wid, None)
                    if a:
                        n = a["n"]; searches[n] += 1
                        imp = table[n]["s"] < a["s_before"] - 1e-6
                        improved_last[n] = imp; streak[n] = streak.get(n, 0) + 1 if imp else 0
                        table[n]["search"] = {"tasks": searches[n], "last": time.strftime("%Y-%m-%d %H:%M:%S"),
                                              "start_s": table[n].get("search", {}).get("start_s", a["s_before"]) if table[n].get("search") else a["s_before"]}
                        wst = workers.get(wid, {})
                        hs = {k: f"{wst.get('hits', {}).get(k, 0)}/{v}" for k, v in wst.get('ends', {}).items() if v}
                        tb = wst.get('task_best')
                        log(f"worker {wid}: n={n} done, {'IMPROVED to %.6f' % table[n]['s'] if imp else 'no improvement'} "
                            f"(task best {('%.6f' % tb) if tb else '–'}, hits/starts {hs or '–'}, searched {searches[n]}x)")
                    workers[wid] = {**workers.get(wid, {}), "state": "waiting"}
                    assign()
                elif kind == "error":
                    log(f"worker {msg[1]} error: {msg[2].strip().splitlines()[-1]}"); print(msg[2], file=sys.stderr)
                elif kind == "found":
                    _, wid, d, m_n, cfg, strat = msg
                    accept(m_n, cfg["cubes"], cfg["s"], f"found by engine search ({strat} strategy, worker {wid})", wid)
            if time.time() - last_import > 30:        # pick up records found by experiment runs
                last_import = time.time()
                for path in glob.glob(os.path.join(ROOT, "results_exp*", "best", f"d{D}_n*.json")):
                    try:
                        e = json.load(open(path))
                        m_n = e["n"]
                        if m_n in table and e.get("cubes") and e["s"] < table[m_n]["s"] - 1e-9 and "removed" not in e.get("source", ""):
                            origin = e.get("source", "")
                            while origin.startswith("found by experiment ("):     # keep only the innermost origin
                                origin = origin[origin.index(": ") + 2:].rstrip(")") if ": " in origin else origin
                            accept(m_n, e["cubes"], e["s"], f"found by experiment ({os.path.basename(os.path.dirname(os.path.dirname(path)))}: {origin})")
                    except Exception as ex:
                        log(f"import of {path} failed: {ex}")
            if time.time() - last_write > 1.0:
                last_write = time.time()
                write_live(args, D, ns, table, workers, events, t_start, searches)
    except KeyboardInterrupt:
        pass
    finally:
        for pr in procs: pr.terminate()
        mgr.shutdown()


def slim(n, e):
    """Trivial grids that the page doesn't draw are sent without their (large) coordinate lists."""
    proved = n == 1 or 2 <= n <= 16 or round(n ** 0.25) ** 4 == n
    if (e.get("source") or "").startswith("grid") and not proved:
        return {k: v for k, v in e.items() if k not in ("cubes", "history")} | {"cubes": e["cubes"][:1] if e.get("cubes") else None}
    return {k: v for k, v in e.items() if k != "history"}


def write_live(args, D, ns, table, workers, events, t_start, searches):
    rate = sum(w.get("steps_per_sec", 0) for w in workers.values() if w.get("state") == "running")
    research = []
    if args.results == "results":           # production: collect the status of experiment / auxiliary runs
        import glob
        for path in glob.glob(os.path.join(ROOT, "results_*", "live.json")):
            try:
                R = json.load(open(path))
                if time.time() - R.get("updated", 0) > 15: continue
                for w, st in R.get("workers", {}).items():
                    if st.get("state") != "running": continue
                    rec = R.get("table", {}).get(str(st.get("n")), {}).get("s")
                    research.append({"dir": os.path.basename(os.path.dirname(path)), "label": R.get("label", ""), "dim": R.get("dim", 4),
                                     "n": st.get("n"), "task_best": st.get("task_best"), "deadline": st.get("deadline"),
                                     "minutes": st.get("minutes"), "record": rec, "strategies": R.get("strategies", "")})
            except Exception:
                pass
    live = {"dim": D, "ns": ns, "mode": "farm", "minutes_per_n": args.minutes, "label": args.label, "strategies": args.strategies,
            "research": research,
            "updated": time.time(), "started": t_start, "uptime": time.time() - t_start,
            "workers": workers, "steps_per_sec": rate, "events": events[-80:], "searches": searches,
            "plan_taken": args.plan_state["taken"] if hasattr(args, "plan_state") else [],
            "control": {k: args.control[k] for k in ("forced", "queue", "blocked")} if hasattr(args, "control") else {},
            "table": {n: slim(n, table[n]) for n in sorted(table)}}
    atomic_json(os.path.join(args.res, "live.json"), live)
    if not args.no_dashboard:
        out = ["\033[H\033[2J", f" 4D tesseract packing (farm) | {rate:,.0f} config-steps/s | http://localhost:{args.port}", "",
               " wkr   n   left   task-best    record"]
        for w, st in sorted(workers.items()):
            if st.get("state") == "running":
                tb = st.get("task_best")
                out.append(f" {w:3d} {st['n']:3d}  {max(0, st.get('deadline', 0) - time.time()):4.0f}s  {('%.6f' % tb) if tb else '    –    ':>10}  {table[st['n']]['s']:.6f}")
            else:
                out.append(f" {w:3d}  {st.get('state')}")
        out.append(""); out += ["  " + e for e in events[-12:]]
        sys.stdout.write("\n".join(out) + "\n"); sys.stdout.flush()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
