"""Structural seeds / constructions proposed by the theorist (2026-10-09).

Packing format as elsewhere: list of {"c": [4], "R": 4x4 (columns = cube axes)}; cube = c + R[-1/2,1/2]^4.

1. conference_rotation(): R = C/sqrt3 with C the order-4 conference matrix (0 on the diagonal, +-1 off it).
   Every one of the 16 diagonal directions sigma=(+-1)^4 lies within one coordinate of +-(a column of C), so a
   cube with this rotation, centred where four gaps of width 1/sqrt3 cross, misses all 16 surrounding
   orthant blocks. That gives s(k^4+1) <= k + 1/sqrt3 (n = 17, 82, 257, ...); certified.
2. k4_plus_one(k, a): the k^4 grid split a | k-a on every axis with gap 1/sqrt3, plus the conference cube.
3. cross_cell_seed(n, s, rng): ROUGH seed (NOT feasible as is) for the 5x5-product frame + "cross cells"
   (the cell with two middle coordinates, one from {x1,x2} and one from {x3,x4}, turned 45 deg in that
   plane). Rebuilt independently from the verbal description; meant to be relaxed by the engine.
4. hybrid17_seed(n, s, rng): ROUGH seed for 18..24: 16 corner cubes + conference centre cube + (n-17) cubes
   taken from the 8 single-plane 45-degree cells of the 5x5 product frame.
5. product_frame_seed(sqA, sqB, extra, s, rng): any 2D x 2D product frame + `extra` cross-cell cubes.
"""
import itertools, math
import numpy as np


def conference_rotation():
    C = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]], float).T / math.sqrt(3)
    C[:, 0] *= -1                       # det +1 (the cube is centrally symmetric, so this changes nothing)
    return C


G4 = 1 / math.sqrt(3)                   # gap the conference cube needs (LP-verified for this R; local numerics
                                        # over all R found nothing smaller than 0.57751 in 2 starts)


def k4_plus_one(k, a=None, g=G4 + 2e-12):
    a = a or k // 2
    pos = [i + 0.5 if i < a else i + 0.5 + g for i in range(k)]
    cubes = [{"c": list(p), "R": np.eye(4).tolist()} for p in itertools.product(pos, repeat=4)]
    cubes.append({"c": [a + g / 2] * 4, "R": conference_rotation().tolist()})
    return k + g, cubes


def plane_rot(i, j, th):
    R = np.eye(4); c, s = math.cos(th), math.sin(th)
    R[i, i] = c; R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R


def _frame25(s):
    lo, hi, mid = 0.5, s - 0.5, s / 2
    sq = [((lo, lo), 0), ((hi, lo), 0), ((lo, hi), 0), ((hi, hi), 0), ((mid, mid), 1)]
    out = []
    for (a, ta), (b, tb) in itertools.product(sq, sq):
        R = np.eye(4)
        if ta: R = R @ plane_rot(0, 1, math.pi / 4)
        if tb: R = R @ plane_rot(2, 3, math.pi / 4)
        out.append({"c": [a[0], a[1], b[0], b[1]], "R": R.tolist()})
    return out


def cross_cells(s):
    """The 16 cross-cell candidates: x_i = x_j = mid (i in {0,1}, j in {2,3}), other two coords at lo/hi,
    turned 45 deg in the (i,j) plane."""
    lo, hi, mid = 0.5, s - 0.5, s / 2
    out = []
    for i in (0, 1):
        for j in (2, 3):
            others = [k for k in range(4) if k not in (i, j)]
            for v in itertools.product((lo, hi), repeat=2):
                c = [0.0] * 4; c[i] = c[j] = mid; c[others[0]], c[others[1]] = v
                out.append({"c": c, "R": plane_rot(i, j, math.pi / 4).tolist()})
    return out


def _jitter(cubes, rng, amp):
    return [{"c": (np.asarray(q["c"]) + amp * rng.standard_normal(4)).tolist(), "R": q["R"]} for q in cubes]


def cross_cell_seed(n, s=2.95, rng=None, amp=0.01):
    rng = rng or np.random.default_rng()
    base = _frame25(s)
    cc = cross_cells(s)
    pick = rng.choice(len(cc), size=max(0, n - 25), replace=False) if n > 25 else []
    cubes = base[:n] + [cc[k] for k in pick]
    return s, _jitter(cubes, rng, amp)


def hybrid17_seed(n, s=2.65, rng=None, amp=0.01):
    rng = rng or np.random.default_rng()
    lo, hi, mid = 0.5, s - 0.5, s / 2
    cubes = [{"c": list(p), "R": np.eye(4).tolist()} for p in itertools.product((lo, hi), repeat=4)]
    cubes.append({"c": [mid] * 4, "R": conference_rotation().tolist()})
    single = [q for q in _frame25(s) if sum(abs(np.asarray(q["c"]) - mid) < 1e-9) == 2]   # 8 one-plane cells
    pick = rng.choice(len(single), size=n - 17, replace=False)
    cubes += [single[k] for k in pick]
    return s, _jitter(cubes, rng, amp)


def product_frame_seed(sqA, sqB, extra, s, rng=None, amp=0.01):
    """sqA, sqB: 2D packings as lists of ((x, y), theta). Adds `extra` cubes at random 'cross' positions:
    one tilted coordinate pair taken from a tilted square of A and one from B, turned 45 deg in a mixed plane."""
    rng = rng or np.random.default_rng()
    out = []
    for (a, ta), (b, tb) in itertools.product(sqA, sqB):
        R = plane_rot(0, 1, ta) @ plane_rot(2, 3, tb)
        out.append({"c": [a[0], a[1], b[0], b[1]], "R": R.tolist()})
    tA = [a for a, t in sqA if abs(t) > 1e-9] or [a for a, _ in sqA]
    tB = [b for b, t in sqB if abs(t) > 1e-9] or [b for b, _ in sqB]
    for _ in range(extra):
        a = tA[rng.integers(len(tA))]; b = tB[rng.integers(len(tB))]
        i, j = int(rng.integers(2)), 2 + int(rng.integers(2))
        c = [*a, *b]
        out.append({"c": c, "R": plane_rot(i, j, math.pi / 4).tolist()})
    return s, _jitter(out, rng, amp)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from geometry import certify
    for k in (2, 3):
        s, cubes = k4_plus_one(k)
        print(len(cubes), s, certify(cubes, s))
