# Centre lemma <=> for all U in O(d) with row-L1 norms rho_j <= 2 and every column k:
#     Psi_k(U) = sum_j |U_jk| (2 - rho_j) <= 1.
# Numerically maximise Psi over O(d) (many random starts, local polish).
import numpy as np, sys
from scipy.linalg import expm
from scipy.optimize import minimize
d = int(sys.argv[1]); rng = np.random.default_rng(1)
iu = np.triu_indices(d, 1)
def U_of(p, U0):
    A = np.zeros((d, d)); A[iu] = p; return U0 @ expm(A - A.T)
def psi(U):
    rho = np.abs(U).sum(1)
    if rho.max() > 2: return -1e9 + 0  # infeasible
    return (np.abs(U) * (2 - rho)[:, None]).sum(0).max()
best = -1; arg = None
for trial in range(4000):
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    if trial % 2: Q = np.eye(d)[rng.permutation(d)] @ expm((lambda A: A - A.T)(np.triu(rng.normal(size=(d,d))*rng.uniform(0,.6),1)))
    f = lambda p: -psi(U_of(p, Q)) + 10 * max(0, np.abs(U_of(p, Q)).sum(1).max() - 2)
    r = minimize(f, np.zeros(len(iu[0])), method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-10, fatol=1e-13))
    v = psi(U_of(r.x, Q))
    if v > best: best = v; arg = U_of(r.x, Q)
print(f"d={d}: max Psi = {best:.10f}  (lemma holds iff <= 1)")
np.set_printoptions(precision=4, suppress=True); print(arg)
