"""Search for packings of n unit tesseracts in a small 4-cube [0,s]^4.

Model: each cube i = centre c_i + R_i [-1/2,1/2]^4 with R_i = expm(skew(p_i)).
Every pair (i,j) carries its own separating hyperplane (w_ij, b_ij); penalty
  sum relu(b - w.v)^2 over vertices of i  +  relu(w.v - b)^2 over vertices of j
is zero  <=>  that hyperplane separates the two cubes.  Container violations are
penalised the same way.  For fixed s we minimise the penalty; s is shrunk
whenever a zero-penalty configuration is found.
"""
import torch, itertools, math, json, sys, time, os
torch.set_default_dtype(torch.float64)
D = int(os.environ.get('DIM', 4))
NP = D * (D - 1) // 2
SIGNS = torch.tensor(list(itertools.product([-0.5, 0.5], repeat=D)))  # 16 x 4
IU = torch.triu_indices(D, D, 1)

def rotations(p):
    A = torch.zeros(p.shape[0], D, D)
    A[:, IU[0], IU[1]] = p
    A = A - A.transpose(1, 2)
    return torch.linalg.matrix_exp(A)

def vertices(c, p):
    R = rotations(p)                       # n x 4 x 4, columns = cube axes
    return c[:, None, :] + SIGNS[None] @ R.transpose(1, 2)   # n x 16 x 4

def penalty(c, p, W, B, s, I, J, margin=0.0):
    V = vertices(c, p)
    w = W / W.norm(dim=1, keepdim=True)
    a = (V[I] * w[:, None, :]).sum(-1)      # P x 16
    b = (V[J] * w[:, None, :]).sum(-1)
    pen = torch.relu(B[:, None] - a + margin).pow(2).sum() + torch.relu(b - B[:, None] + margin).pow(2).sum()
    pen = pen + torch.relu(V - s + margin).pow(2).sum() + torch.relu(-V + margin).pow(2).sum()
    return pen

def init_pairs(c, p, I, J):
    W = (c[I] - c[J]) + 1e-3 * torch.randn(len(I), D)
    w = W / W.norm(dim=1, keepdim=True)
    B = ((c[I] + c[J]) / 2 * w).sum(1)
    return W, B

def relax(c, p, W, B, s, I, J, steps=3000, lr=0.01, margin=0.0):
    params = [c, p, W, B]
    for t in params: t.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    for k in range(steps):
        opt.zero_grad()
        L = penalty(c, p, W, B, s, I, J, margin)
        if L.item() < 1e-20: break
        L.backward(); opt.step()
    # polish with LBFGS
    opt = torch.optim.LBFGS(params, lr=1, max_iter=500, tolerance_grad=1e-14, tolerance_change=1e-16, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); L = penalty(c, p, W, B, s, I, J, margin); L.backward(); return L
    for _ in range(4): opt.step(closure)
    with torch.no_grad():
        return penalty(c, p, W, B, s, I, J, margin).item()

def search(n, s0, seed=0, shrink=0.995, tries=60, verbose=True):
    torch.manual_seed(seed)
    I, J = torch.triu_indices(n, n, 1)
    s = s0
    c = torch.rand(n, D) * (s - 1) + 0.5
    p = torch.randn(n, NP) * 0.3
    W, B = init_pairs(c, p, I, J)
    best = None
    fails = 0
    while fails < tries:
        L = relax(c, p, W, B, s, I, J, margin=1e-6)
        if L < 1e-14:
            best = (s, c.detach().clone(), p.detach().clone())
            if verbose: print(f"  n={n} feasible at s={s:.6f}", flush=True)
            fails = 0
            s_new = 1 + (s - 1) * shrink
            with torch.no_grad():   # rescale centres toward the new box
                c.mul_((s_new - 1) / (s - 1)).sub_(0.5 * (s_new - 1) / (s - 1) - 0.5)
            s = s_new
        else:
            fails += 1
            with torch.no_grad():   # shake
                c.add_(torch.randn_like(c) * 0.02); p.add_(torch.randn_like(p) * 0.05)
    return best

if __name__ == "__main__":
    n = int(sys.argv[1]); s0 = float(sys.argv[2]); seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    results = []
    for seed in range(seeds):
        t0 = time.time()
        b = search(n, s0, seed=seed)
        if b:
            s, c, p = b
            print(f"seed {seed}: s = {s:.6f}  ({time.time()-t0:.0f}s)", flush=True)
            results.append(dict(n=n, s=s, centers=c.tolist(), params=p.tolist(), seed=seed))
        else:
            print(f"seed {seed}: nothing feasible from s0={s0}", flush=True)
    results.sort(key=lambda r: r["s"])
    json.dump(results, open(f"results/d{D}_n{n}.json", "w"))
