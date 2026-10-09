# Independent overlap check (no LP): (1) orthonormality, (2) all 16 vertices inside [0,s]^4,
# (3) for every pair, dense random points of cube i (interior + surface) tested against cube j's
#     8 facet inequalities |R_j^T (x - c_j)|_inf < 1/2 - eps  -> any hit = overlap.
import json, sys, itertools, numpy as np
e = json.load(open(sys.argv[1])); s = e["s"]; cubes = e["cubes"]
C = np.array([q["c"] for q in cubes]); R = np.array([q["R"] for q in cubes]); n = len(C)
print("n", n, "s", s)
print("max |R^T R - I|", max(np.abs(r.T @ r - np.eye(4)).max() for r in R))
sig = np.array(list(itertools.product([-.5, .5], repeat=4)))
V = C[:, None, :] + sig @ R.transpose(0, 2, 1)
print("container: min vertex coord", V.min(), " max - s", V.max() - s)
rng = np.random.default_rng(0); worst = -1
for i in range(n):
    T = rng.uniform(-.5, .5, size=(200000, 4))
    k = rng.integers(0, 4, size=len(T)); T[np.arange(len(T))[:100000], k[:100000]] = np.sign(T[np.arange(100000), k[:100000]]) * .5  # half on facets
    X = C[i] + T @ R[i].T
    for j in range(n):
        if i == j: continue
        local = (X - C[j]) @ R[j]           # coordinates in cube j's frame
        depth = 0.5 - np.abs(local).max(1)  # >0 means inside cube j
        worst = max(worst, depth.max())
print("max penetration depth of sampled points of one cube into another:", worst, "(<= 1e-9 means no overlap found)")
# minimum centre distance
D = np.linalg.norm(C[:, None] - C[None], axis=-1) + np.eye(n) * 9
print("min centre distance", D.min())
