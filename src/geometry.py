"""Exact-ish geometry helpers (numpy/scipy): vertices, certification, known constructions.

A packing is a list of cubes {"c": [d floats], "R": d x d (columns = cube axes)};
cube = { c + R t : t in [-1/2, 1/2]^d }.  Container = [0, s]^d.
"""
import itertools, math
import numpy as np
from scipy.optimize import linprog


def signs(d):
    return np.array(list(itertools.product([-0.5, 0.5], repeat=d)))


def cube_vertices(c, R):
    c = np.asarray(c, float); R = np.asarray(R, float)
    return c[None, :] + signs(len(c)) @ R.T


def separation(Va, Vb):
    """Max delta s.t. some hyperplane w.x=b (|w|_inf<=1) has w.Va >= b+delta, w.Vb <= b-delta.
    delta >= 0  <=>  the two convex hulls have disjoint interiors."""
    d = Va.shape[1]
    # variables: w (d), b, delta ; maximise delta
    A = np.vstack([np.hstack([-Va, np.ones((len(Va), 1)), np.ones((len(Va), 1))]),
                   np.hstack([Vb, -np.ones((len(Vb), 1)), np.ones((len(Vb), 1))])])
    r = linprog(np.r_[np.zeros(d + 1), -1.0], A_ub=A, b_ub=np.zeros(len(A)),
                bounds=[(-1, 1)] * d + [(None, None), (None, 1.0)], method="highs")
    return -r.fun if r.status == 0 else -np.inf


def certify(cubes, s, tol=1e-9):
    """Returns (ok, worst_container_violation, worst_pair_separation)."""
    V = [cube_vertices(q["c"], q["R"]) for q in cubes]
    d = V[0].shape[1] if V else 0
    for q in cubes:  # orthonormality
        R = np.asarray(q["R"]);
        if np.abs(R.T @ R - np.eye(d)).max() > 1e-9:
            return False, np.inf, -np.inf
    cont = max(max(v.max() - s, -v.min()) for v in V) if V else 0.0
    worst = np.inf
    C = [np.asarray(q["c"]) for q in cubes]
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if np.linalg.norm(C[i] - C[j]) > math.sqrt(d) + 1e-9:  # circumradius sqrt(d)/2 each
                continue
            worst = min(worst, separation(V[i], V[j]))
    ok = cont <= tol and worst >= -tol
    return ok, float(cont), float(worst)


# ---------------------------------------------------------------- constructions
def grid_packing(n, d):
    k = 1
    while k ** d < n: k += 1
    cells = list(itertools.product(range(k), repeat=d))
    cells.sort(key=lambda t: t[::-1])   # last axis (shown as time) slowest: fill whole 3D layers first
    cells = cells[:n]
    return float(k), [{"c": [x + 0.5 for x in cell], "R": np.eye(d).tolist()} for cell in cells]


def five_square():
    """Optimal 5 unit squares in side 2 + 1/sqrt2 (4 corners + one at 45 degrees)."""
    s = 2 + 1 / math.sqrt(2)
    sq = [((0.5, 0.5), 0.0), ((s - .5, .5), 0.0), ((.5, s - .5), 0.0), ((s - .5, s - .5), 0.0), ((s / 2, s / 2), math.pi / 4)]
    return s, sq


def rot2(th):
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def product_packing(A, B):
    """Product of a packing A (squares in [0,s]^2) with B (squares in [0,s]^2) -> 4D."""
    out = []
    for (ca, ta) in A:
        for (cb, tb) in B:
            R = np.zeros((4, 4)); R[:2, :2] = rot2(ta); R[2:, 2:] = rot2(tb)
            out.append({"c": [ca[0], ca[1], cb[0], cb[1]], "R": R.tolist()})
    return out


def lift_3d(cfg3, layers=2):
    """3D packing x [0, layers]: stack `layers` copies along the 4th axis."""
    out = []
    for k in range(layers):
        for q in cfg3["cubes"]:
            R = np.eye(4); R[:3, :3] = np.asarray(q["R"])
            out.append({"c": list(q["c"]) + [k + 0.5], "R": R.tolist()})
    return max(cfg3["s"], float(layers)), out


def product_2d(A, B):
    """Product of two 2D packings (dicts with s, cubes having 2x2 R) -> 4D packing in side max(sA, sB)."""
    out = []
    for qa in A["cubes"]:
        for qb in B["cubes"]:
            R = np.zeros((4, 4)); R[:2, :2] = np.asarray(qa["R"]); R[2:, 2:] = np.asarray(qb["R"])
            out.append({"c": list(qa["c"]) + list(qb["c"]), "R": R.tolist()})
    return max(A["s"], B["s"]), out


def load_aux(root):
    """Best 2D / 3D packings found by the engine (results_aux2, results_aux3), used for 4D constructions."""
    import json, os, glob
    aux = {}
    for d in (2, 3):
        for path in glob.glob(os.path.join(root, f"results_aux{d}", "best", f"d{d}_n*.json")):
            e = json.load(open(path))
            if e.get("cubes"): aux[(d, e["n"])] = e
    return aux


def known_constructions_4d(nmax=81, root=None):
    """Best explicit constructions we can write down, per n (n -> dict)."""
    res = {}
    s5, sq5 = five_square()
    sq5_list = sq5
    prod25 = product_packing(sq5, sq5)
    # order so that subsets keep the nice structure: corner-corner first
    prod25.sort(key=lambda q: (abs(np.asarray(q["R"])).sum() > 4.0 + 1e-9, q["c"]))
    for n in range(1, nmax + 1):
        s, cubes = grid_packing(n, 4)
        entry = {"n": n, "s": s, "cubes": cubes, "source": "grid (trivial)", "certified": True}
        if 17 <= n <= 25 and s5 < s:
            entry = {"n": n, "s": s5, "cubes": prod25[:n],
                     "source": "product of two optimal 5-square packings (2+1/sqrt2)^2", "certified": True}
        res[n] = entry
    # constructions from our own certified 2D/3D packings (if present)
    if root:
        aux = load_aux(root)
        sq5 = {"s": s5, "cubes": [{"c": list(c), "R": rot2(t).tolist()} for c, t in sq5_list]}
        twod = {5: sq5}
        for (d, m), e in aux.items():
            if d == 2: twod[m] = e
        cands = []
        for a, A in twod.items():           # 2D x 2D products
            for b, B in twod.items():
                if a <= b: cands.append((a * b, *product_2d(A, B), f"product of {a}-square and {b}-square packings"))
        for (d, m), e in aux.items():       # 3D x [0,2] lifts
            if d == 3: cands.append((2 * m, *lift_3d(e, 2), f"{m}-cube 3D packing (found by our search) x [0,2]"))
        for cnt, s_c, cubes, why in cands:
            for n in range(1, min(cnt, nmax) + 1):
                if s_c < res[n]["s"] - 1e-12:
                    res[n] = {"n": n, "s": s_c, "cubes": cubes[:n], "source": why, "certified": True, "construction": True}
    # bounds cited without coordinates: product of Friedman's 3D packings with a 1D segment of 2
    cited = {26: (2.956, "Friedman 3D 13-cube packing x [0,2]"),
             27: (2 + 7 * math.sqrt(2) / 10, "Friedman 3D 14-cube packing x [0,2] (14x2=28)"),
             28: (2 + 7 * math.sqrt(2) / 10, "Friedman 3D 14-cube packing x [0,2]")}
    for n, (s, why) in cited.items():
        if s < res[n]["s"]:
            alt = {k: res[n][k] for k in ("s", "cubes", "source")} if res[n].get("cubes") else None
            res[n] = {"n": n, "s": s, "cubes": None, "source": why + " — coordinates not reproduced",
                      "certified": False, "cited": True, "alt": alt}
    return res


def lower_bound_4d(n):
    """Rigorous lower bounds we use (volume; s>=2 for n>=2 pending the centre lemma)."""
    return max(n ** 0.25, 1.0)
