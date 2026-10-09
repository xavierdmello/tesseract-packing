"""Check the native kernel against the torch kernel: penalty, gradients, and an Adam trajectory."""
import sys, time, torch, numpy as np
sys.path.insert(0, "src")
from kernel import Batch
import nkernel
torch.set_default_dtype(torch.float64); torch.manual_seed(0)

def make(n, B, s=2.8, r=0.0):
    m = Batch(4, n, B, "cpu", torch.float64)
    m.s.fill_(s); m.r.fill_(r)
    m.c.copy_(torch.rand(B, n, 4) * (s - 1) + 0.5); m.p.copy_(m.random_p(B, n, spread=0.6))
    m.reset_planes(torch.arange(B)); m.W.add_(0.3 * torch.randn_like(m.W)); m.Bh.add_(0.1 * torch.randn_like(m.Bh))
    return m

worst = 0
for n, r in ((5, 0.0), (17, 0.0), (17, 0.2), (30, 0.1)):
    m = make(n, 1, r=r)
    for t in m.params(): t.requires_grad_(True)
    pen, mx = m.violation(); pen.sum().backward()
    g_t = [t.grad[0].clone() for t in m.params()]
    with torch.no_grad():
        gs = [torch.zeros_like(t[0]) for t in m.params()]
        pn = nkernel._lib.grad_one(n, *[nkernel.ptr(t[0].detach().contiguous()) for t in m.params()],
                                   float(m.s[0]), float(m.r[0]), m.margin, 0, *[nkernel.ptr(g) for g in gs])
    rel = max(float((a - b).abs().max() / (a.abs().max() + 1e-12)) for a, b in zip(g_t, gs))
    worst = max(worst, rel)
    print(f"n={n:3d} r={r}: penalty torch {pen.item():.10e} native {pn:.10e}  max rel grad diff {rel:.2e}")
print("GRADIENTS", "OK" if worst < 1e-9 else "MISMATCH")

# trajectory: 50 Adam steps from the same start
m1 = make(17, 4); m2 = make(17, 4)
for a, b in zip(m2.params(), m1.params()): a.copy_(b)
for t in m1.params(): t.requires_grad_(True)
opt = torch.optim.Adam(m1.params(), lr=0.01)
for _ in range(50):
    opt.zero_grad(); pen, _ = m1.violation(); pen.sum().backward(); opt.step()
na = nkernel.NativeAdam(m2, 0.01, prune=False); pn, mxn = na.steps(50)
with torch.no_grad(): pt, mxt = m1.violation()
print("trajectory max |dc|", float((m1.c - m2.c).abs().max()), " penalties", pt.tolist()[:2], pn.tolist()[:2])
