"""Batched packing optimiser (shared by engine workers and the benchmark).

B independent configurations of n unit d-cubes are optimised in one set of
tensors.  Pair (i,j) carries a free separating hyperplane (w, b); penalty is the
squared hinge of vertex violations plus container violations (zero <=> valid).
"""
import itertools, time
import torch


class Batch:
    def __init__(self, D, n, B, device="cpu", dtype=torch.float64):
        self.D, self.n, self.B = D, n, B
        self.dev, self.dt = torch.device(device), dtype
        kw = dict(device=self.dev, dtype=dtype)
        self.kw = kw
        self.SIG = torch.tensor(list(itertools.product([-0.5, 0.5], repeat=D)), **kw)
        self.NP = 8 if D == 4 else D * (D - 1) // 2
        self.IU = torch.triu_indices(D, D, 1, device=self.dev)
        self.I, self.J = torch.triu_indices(n, n, 1, device=self.dev)
        self.P = self.I.numel()
        self.margin = 1e-7 if dtype == torch.float64 else 2e-5
        self.s = torch.zeros(B, **kw)
        self.c = torch.zeros(B, n, D, **kw)
        self.p = torch.zeros(B, n, self.NP, **kw)
        self.W = torch.zeros(B, self.P, D, **kw)
        self.Bh = torch.zeros(B, self.P, **kw)
        self.r = torch.zeros(B, **kw)    # rounding radius: body = (1-2r)-cube (+) ball(r); r=0.5 ball, r=0 cube

    # ---------------- rotations
    @staticmethod
    def _L(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([torch.stack([w, -x, -y, -z], -1), torch.stack([x, w, -z, y], -1),
                            torch.stack([y, z, w, -x], -1), torch.stack([z, -y, x, w], -1)], -2)

    @staticmethod
    def _R(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([torch.stack([w, -x, -y, -z], -1), torch.stack([x, w, z, -y], -1),
                            torch.stack([y, -z, w, x], -1), torch.stack([z, y, -x, w], -1)], -2)

    def rot(self, p):
        if self.D == 4:
            a = p[..., :4]; b = p[..., 4:]
            a = a / a.norm(dim=-1, keepdim=True); b = b / b.norm(dim=-1, keepdim=True)
            return self._L(a) @ self._R(b)
        A = torch.zeros(*p.shape[:-1], self.D, self.D, **self.kw)
        A[..., self.IU[0], self.IU[1]] = p
        return torch.linalg.matrix_exp(A - A.transpose(-1, -2))

    def random_p(self, k, n, spread=0.3, near_frac=0.6):
        if self.D == 4:
            a = torch.randn(k, n, 4, **self.kw); b = torch.randn(k, n, 4, **self.kw)
            near = torch.rand(k, n, 1, **self.kw) < near_frac
            e = torch.zeros(4, **self.kw); e[0] = 1
            return torch.cat([torch.where(near, e + spread * a, a), torch.where(near, e + spread * b, b)], -1)
        return torch.randn(k, n, self.NP, **self.kw) * spread

    # ---------------- objective
    def violation(self):
        R = self.rot(self.p)
        SIGr = self.SIG[None] * (1 - 2 * self.r)[:, None, None]             # B 2^D D (shrunken core cube)
        V = self.c[:, :, None, :] + SIGr[:, None] @ R.transpose(-1, -2)     # B n 2^D D
        w = self.W / self.W.norm(dim=-1, keepdim=True)
        a = (V[:, self.I] * w[:, :, None, :]).sum(-1)
        b = (V[:, self.J] * w[:, :, None, :]).sum(-1)
        m = self.margin + self.r[:, None, None]          # rounded bodies need clearance r on each side
        va = torch.relu(self.Bh[..., None] - a + m)
        vb = torch.relu(b - self.Bh[..., None] + m)
        sc = self.s[:, None, None, None]
        mc = m[..., None]
        vc = torch.relu(V - sc + mc) + torch.relu(-V + mc)
        pen = va.pow(2).sum((1, 2)) + vb.pow(2).sum((1, 2)) + vc.pow(2).sum((1, 2, 3))
        mx = torch.maximum(torch.maximum(va.amax((1, 2)), vb.amax((1, 2))), vc.amax((1, 2, 3)))
        return pen, mx

    def reset_planes(self, idx):
        I, J = self.I, self.J
        Wd = (self.c[idx][:, I] - self.c[idx][:, J]) + 1e-3 * torch.randn(len(idx), self.P, self.D, **self.kw)
        Wd = Wd / Wd.norm(dim=-1, keepdim=True)
        self.W[idx] = Wd
        self.Bh[idx] = ((self.c[idx][:, I] + self.c[idx][:, J]) / 2 * Wd).sum(-1)

    def params(self):
        return [self.c, self.p, self.W, self.Bh]

    def config(self, k):
        R = self.rot(self.p[k])
        return {"s": float(self.s[k]), "cubes": [{"c": self.c[k, i].double().tolist(), "R": R[i].double().tolist(),
                                                   "p": self.p[k, i].double().tolist()} for i in range(self.n)]}


def fit_p(R_list, iters=800):
    """Quaternion-pair parameters p (8 numbers) with L(a)R(b) = R, for a list of 4x4 rotations (CPU, float64)."""
    m = Batch(4, 2, 1, "cpu", torch.float64)
    R = torch.tensor(R_list, dtype=torch.float64)
    if torch.allclose(R, torch.eye(4, dtype=torch.float64).expand_as(R), atol=1e-14):   # unrotated: closed form
        return [[1.0, 0, 0, 0, 1.0, 0, 0, 0] for _ in R_list], 0.0
    if torch.linalg.det(R).min() < 0:   # reflections: flip one axis (cube is symmetric, so this is the same cube)
        R = R.clone(); neg = torch.linalg.det(R) < 0; R[neg, :, 0] *= -1
    k = R.shape[0]; best = None
    for trial in range(6):
        q = (torch.tensor([1., 0, 0, 0, 1, 0, 0, 0]).repeat(k, 1) if trial == 0 else torch.randn(k, 8, dtype=torch.float64))
        q = (q + 0.01 * torch.randn(k, 8, dtype=torch.float64)).requires_grad_(True)
        opt = torch.optim.LBFGS([q], max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-15, tolerance_change=1e-18)
        def cl():
            opt.zero_grad(); L = (m.rot(q) - R).pow(2).sum(); L.backward(); return L
        opt.step(cl)
        with torch.no_grad():
            err = (m.rot(q) - R).pow(2).sum((1, 2))
            if best is None: best = [err.clone(), q.detach().clone()]
            else:
                b = err < best[0]; best[0][b] = err[b]; best[1][b] = q.detach()[b]
        if best[0].max() < 1e-20: break
    return best[1].tolist(), float(best[0].max())


def bench(D, n, B, device, dtype, seconds=8.0, threads=1):
    torch.set_num_threads(threads)
    m = Batch(D, n, B, device, dtype)
    m.s.fill_(n ** (1 / D) + 0.8)
    m.c.copy_(torch.rand(B, n, D, **m.kw) * (m.s[:, None, None] - 1) + 0.5)
    m.p.copy_(m.random_p(B, n))
    m.reset_planes(torch.arange(B, device=m.dev))
    for t in m.params(): t.requires_grad_(True)
    opt = torch.optim.Adam(m.params(), lr=0.01)
    steps = 0
    def sync():
        if m.dev.type == "mps": torch.mps.synchronize()
    for _ in range(5):
        opt.zero_grad(); pen, _ = m.violation(); pen.sum().backward(); opt.step()
    sync(); t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(10):
            opt.zero_grad(); pen, _ = m.violation(); pen.sum().backward(); opt.step()
        sync(); steps += 10
    return steps * B / (time.time() - t0)


if __name__ == "__main__":
    import sys
    D, n, B, dev, dt = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
    thr = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    r = bench(D, n, B, dev, torch.float64 if dt == "f64" else torch.float32, threads=thr)
    print(f"{dev} {dt} n={n} B={B} threads={thr}: {r:,.0f} config-steps/s")
