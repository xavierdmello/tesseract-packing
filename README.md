# Tesseracts in Tesseracts

Packing n unit 4-cubes (tesseracts) into the smallest possible 4-cube, the 4D analogue of
Friedman's *Squares in Squares* and *Cubes in Cubes*. As of 2026-10-09 we found no published 4D results.

## Run

```bash
.venv/bin/python src/engine.py --ns 17-100 --minutes 1     # all cores, one n at a time
open http://localhost:8765                                   # live website
.venv/bin/python src/serve.py                                # website on saved results only
```

Options: `--dim 3` (validation in 3D), `--results DIR`, `--strategies best,grow,shrink,random,morph`,
`--cpu-workers 11 --gpu-workers 1`.

## Method

* **Model.** Each configuration has n cubes (centre + rotation; 4D rotations are stored as a quaternion pair) and one free separating
  hyperplane per pair. The penalty is the squared hinge loss of vertex violations plus container violations. It is zero exactly when
  the configuration is a valid packing.
* **Search.** Workers optimize batches of configurations at once with vectorized Adam. Each configuration shrinks the box
  whenever it is valid and backs off when it jams, with an adaptive step size.
* **Strategies used for restarts.** Each worker has a different mix of:
  * jitter the best packing;
  * best (n−1) packing plus one cube;
  * best (n+1) packing minus one cube;
  * random start;
  * blob→cube morph, after Yohei Nakajima: compress balls, then morph through rounded cubes to cubes.
* **Collaboration.** Workers share a "best so far" board. The main process certifies every improvement and propagates it to
  smaller n by deleting cubes, since s(k) ≤ s(n) for k < n.
* **Certification.** The engine checks orthonormality and that every vertex lies in [0,s]⁴. For every pair of cubes it solves a
  linear program for a separating hyperplane with |w|∞ ≤ 1 and requires separation ≥ −1e−10. A tiny residual overlap is fixed by
  spreading the centres by a factor (1+λ) and re-measuring s. `proofs/independent_check.py` is a second check by point sampling
  that does not use the LP.
* **Hardware.** M3 Pro with 11 CPU processes plus 1 Metal (MPS) process. PyTorch's per-operation overhead dominates, not
  arithmetic: float32 and float64 run at the same speed on the CPU.

## Results so far

| n | best s | notes |
|---|---|---|
| 1 | 1 | trivial |
| 2–16 | 2 | grid; lower bound via the centre lemma (see below) |
| 17 | **2.578422** | engine search; beats the product construction 2+1/√2 |
| 18–25 | 2.707107 | product of two optimal 5-square packings (2+1/√2)² |
| 26 | ≤ 2.956 | Friedman's 3D 13-cube packing × [0,2] (cited) |
| 27–28 | ≤ 2.98995 | Friedman's 3D 14-cube packing × [0,2] (cited) |

`results/best/d4_n*.json` holds the current best packing for each n, with coordinates, rotations and certificate.

## Lower bounds

* Volume: s(n) ≥ n^{1/4}.
* **Centre lemma.** A unit d-cube inside a d-cube of side < 2 contains the centre, so s(n) ≥ 2 for n ≥ 2. Using the facet
  argument plus an LP over the cube's centre, this is equivalent to the following: for every U ∈ O(d) and every column k,
  Σ_j |U_jk|(2 − ρ_j) ≤ 1, where ρ_j is the L1 norm of row j. The numerical maximum is exactly 1 in d = 2, 3, 4, 5
  (`proofs/center_lemma_numeric.py`). A rigorous proof for d = 4 is still open (the inequality is tight to second order at
  signed permutation matrices).
