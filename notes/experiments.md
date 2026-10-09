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

**E5 outcome:** invalid. The counters were reset when the engine started its next task, so no readout. The logging was fixed (per-task hits/starts now go to events.log).

## E6: Fair morph test v2 (slow, annealed), 2D n = 11, 10 min, 1 core each (machine at full load) — 14:14
- **Morph v2 settings:** `--morph-k 100,200 --morph-noise 0.02` (10k–20k steps of morphing, annealing jiggle).
- **Result:**

  | arm | best s | starts within 0.1% of 3.8771 |
  |---|---|---|
  | morph v2 | **3.8865** | 0 / 74 |
  | random (rigid) | 3.8877 | 0 / 107 |
- **Conclusion:** first time morph beat its control, slightly. Neither hit the threshold. Inconclusive. Morph stays as an exploration share; the theorist agent may revisit it.

## E7: Native C kernel — 14:22
- **Correctness:** the penalty and all gradients match torch autograd to ≤ 3e-14 relative error. A 50-step Adam trajectory matches to 7e-16.
- **Speed** (1 core each, torch and native run side by side, machine at full load; config-steps/s):

  | n | torch | native | native + skip far pairs |
  |---|---|---|---|
  | 17 | 2,794 | 74,685 | 109,649 |
  | 30 | 786 | 19,191 | 40,822 |
  | 82 | 129 | 3,172 | 5,380 |
- **Validation run:** 2 cores, from scratch, 1.5 min. Reached n = 17: 2.5792 and n = 26: 2.9476, which beat production's 26.
- **Decision:** production switched to `--native` at 14:26. Production throughput went from ~17k to ~398k config-steps/s. Within 2 minutes it took n = 28 below the 2.98995 bound and n = 29 below s = 3.
