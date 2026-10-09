# Numerically test: can a unit d-cube fit in [-t,t]^d with t<1 while NOT containing the origin in its interior?
# For fixed rotation U (rows u_k), and fixed facet k with u_k.c >= 1/2, minimizing t = max_j(|c_j|+h_j) over c is an LP.
import numpy as np, sys
from scipy.optimize import linprog, minimize
from scipy.linalg import expm
d = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rng = np.random.default_rng(0)
iu = np.triu_indices(d, 1)
def rot(p):
    A = np.zeros((d, d)); A[iu] = p; A -= A.T; return expm(A)
def best_t(U):
    h = 0.5 * np.abs(U).sum(0)              # half-width along coordinate j
    best = np.inf
    for k in range(d):
        # vars c (d), t ; min t s.t. c_j + h_j <= t, -c_j + h_j <= t, u_k.c >= 1/2
        A = []; b = []
        for j in range(d):
            e = np.zeros(d + 1); e[j] = 1; e[d] = -1; A.append(e); b.append(-h[j])
            e = np.zeros(d + 1); e[j] = -1; e[d] = -1; A.append(e); b.append(-h[j])
        e = np.zeros(d + 1); e[:d] = -U[k]; A.append(e); b.append(-0.5)
        r = linprog(np.r_[np.zeros(d), 1], A_ub=np.array(A), b_ub=b, bounds=[(None, None)] * (d + 1))
        best = min(best, r.fun)
    return best
best = np.inf
for trial in range(300):
    p0 = rng.normal(size=len(iu[0])) * rng.uniform(0, 1.5)
    r = minimize(lambda p: best_t(rot(p)), p0, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-9, fatol=1e-12))
    if r.fun < best:
        best = r.fun; print(trial, best, flush=True)
print("d=%d  min t for a unit cube avoiding the origin: %.6f" % (d, best))
