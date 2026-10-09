# Experiment log

Each entry gives the hypothesis, the setup, the result and the decision. All times are local, 2026-10-09.

## E1: CPU vs GPU throughput (PyTorch kernel) — 12:45
- **Setup:** `src/kernel.py` benchmark, 4D, n = 17 and 30, 8 s per configuration.
- **Result (config-steps/s):**

  | | n = 17 | n = 30 |
  |---|---|---|
  | 12 CPU processes | 59k | 21k |
  | GPU (MPS), batch 1024–4096 | 44k | 15k |

  f32 runs at the same speed as f64 on the CPU, so the kernel is overhead-bound.
- **Caveat:** CPU + GPU together was never measured cleanly; orphaned processes contaminated the only reading.
- **Decision:** production is CPU-only (user decision, 13:50). Revisit with a native kernel.

## E2: Rigid vs morph (v0, first version), 3D n = 12, 3 min, 12 workers each — 13:19
- **Result:**

  | arm | best s |
  |---|---|
  | rigid (best/grow/shrink/random) | **2.9355** |
  | morph v0 | 2.9987 |
- **Flaw:** the rigid arm had seeded strategies and the morph arm didn't. This is not a fair comparison.

## E3: Morph v1 (compress → morph → rigid), 3D n = 12, 3 min — 13:26
- **Result:** best s 2.9620. Only 1 of 12 workers got below 3.0.
- **Same flaw as E2.** Also, the morph schedule (800–4,000 steps) is much shorter than Yohei's (~15,000).
- **Decision:** morph kept at a small exploration share. A fair test is needed → E5.

## E4: Engine sanity in 2D — 13:29
- 2D, n = 10 → 3.707115 (optimum 3+1/√2 = 3.707107) in < 1 min.
- 2D, n = 11 → 3.879404 (best known 3.8771).
- **Conclusion:** the engine finds near-optimal 2D packings quickly.

## E5: Fair morph test, 2D n = 11 (Yohei's benchmark) — running since 14:01
- **Arms:** `--strategies random` vs `--strategies morph`. 1 core each, 10 min, both starting from scratch.
- **Metric:** fraction of finished starts reaching s ≤ 3.8810 (within 0.1% of 3.8771), plus best s.
- **Promotion rule:** promote morph in production if it reaches hits faster, or finds structures that random starts don't.
