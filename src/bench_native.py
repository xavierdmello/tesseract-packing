import sys, time, torch
sys.path.insert(0, "src")
from kernel import Batch
import nkernel
torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(0)
mode, n, B, secs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
m = Batch(4, n, B, "cpu", torch.float64)
s = n ** 0.25 + 0.8; m.s.fill_(s)
m.c.copy_(torch.rand(B, n, 4) * (s - 1) + 0.5); m.p.copy_(m.random_p(B, n)); m.reset_planes(torch.arange(B))
steps = 0; t0 = time.time()
if mode == "torch":
    for t in m.params(): t.requires_grad_(True)
    opt = torch.optim.Adam(m.params(), lr=0.01)
    while time.time() - t0 < secs:
        for _ in range(10): opt.zero_grad(); pen, _ = m.violation(); pen.sum().backward(); opt.step()
        steps += 10
else:
    na = nkernel.NativeAdam(m, 0.01, prune=(mode == "native_prune"))
    while time.time() - t0 < secs:
        na.steps(10); steps += 10
print(f"{mode:13s} n={n:3d}: {steps * B / (time.time() - t0):10,.0f} config-steps/s")
