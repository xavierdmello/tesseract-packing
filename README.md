# tesseract-packing

What is the smallest 4-cube that holds n unit tesseracts (4-dimensional cubes), with rotations allowed?
This is the 4D analogue of Erich Friedman's *Squares in Squares* and *Cubes in Cubes*. This repository contains a
search engine, certified packings, a live website, and the research notes behind them.

Branches:
- **`research`**: everything (engine, notes, plan, results).
- **`site`**: the public website, published with GitHub Pages and updated periodically from `research`.

## Results (best certified side s found here; work in progress)

| n | s (ours) | note |
|---|---|---|
| 1 | 1 | trivial |
| 2–16 | 2 | grid. Lower bound via the centre lemma (numerically verified, see below) |
| 17 | **2.577350** ≈ 2 + 1/√3 | the 16-grid plus one tilted cube. Beats the product construction 2 + 1/√2 = 2.70711 |
| 18–25 | 2.707107 | product of two optimal 5-square packings (2 + 1/√2)² |
| 26–29 | 2.9433–2.9937 | grown from the 25-cube product. **Better values exist in [hockyy/tesseract-packing](https://github.com/hockyy/tesseract-packing): ≈2.9428 for 26–29** |
| 30–80 | 3 | trivial grid here. hockyy reports n = 30 at 2.98957 |
| 81 | 3 | optimal by volume |
| 82 | **3.577350** ≈ 3 + 1/√3 | the 81-grid plus one tilted cube (same family as n = 17). Beats the 10×10 product 3.70711 |
| 83–100 | 3.707107 | product of two 10-square packings |
| 101 | **3.8523** | grown from the 10×10 product. Beats the 11×11 product 3.87940 |
| 102–121 | 3.879404 | product of two 11-square packings |

Conjecture (from n = 17 and 82): s(k⁴ + 1) ≤ k + 1/√3.

`results/best/d4_n*.json` holds every packing: centre and rotation per cube, the certificate, and how it was found.
A cube is `c + R·[-1/2, 1/2]^4`, where the columns of R are the cube's axes; the box is `[0, s]^4`.

## Related work

- **[hockyy/tesseract-packing](https://github.com/hockyy/tesseract-packing)** (Oct 9, 2026, concurrent and independent; live page at packing.mikira.id). Same problem. They report n = 26–29 at ≈2.9428 (26 conjecturally 2 + 2√2/3) and n = 30 at 2.98957. The structure is the 25-cube product frame with extra cubes in "cross cells". Those are better than our values for 26–30.
- **[yoheinakajima/soft-to-rigid-packing](https://github.com/yoheinakajima/soft-to-rigid-packing)** (Oct 8, 2026): the ball → rounded cube → cube "morph" method, with a 3D record for n = 12 (2.93152). Our engine has a morph strategy inspired by it.
- Erich Friedman, [Cubes in Cubes](https://erich-friedman.github.io/packing/cubincub/) and the [Squares in Squares survey](https://www.combinatorics.org/ojs/index.php/eljc/article/view/DS7). These are the 2D/3D baselines used for products and lifts.
- Januszewski & Zielonka (2024), *Packing of non-blocking four-dimensional cubes into the unit cube*, and Meir–Moser (1968). These concern a different problem: volume guarantees for upright cubes of different sizes.

We found no published table for this 4D problem before Oct 2026. These are search results, not optimality proofs.

## How it works

- **Model.** Each cube has a centre and a rotation (stored as a quaternion pair). Each pair of cubes has its own separating hyperplane. The penalty is the squared hinge loss of vertex violations plus container violations, and it is zero exactly when the packing is valid.
- **Engine** (`src/engine.py`). It runs in "farm" mode: each CPU core searches its own n for a time set in `plan.json`, and all cores share a best-so-far board. Each restart picks a strategy:
  - **best:** jitter the best packing for n;
  - **grow:** best packing for n − 1 plus one cube;
  - **shrink:** best packing for n + 1 minus one cube;
  - **random:** a fresh random start;
  - **morph:** start from balls and harden them into cubes.
- **Native kernel** (`src/native/kernel4d.c`). Penalty, analytic gradient and Adam in C. It is checked against PyTorch autograd to 1e−14 and runs 25–50× faster than the PyTorch version.
- **Certification** (`src/geometry.py`). Every vertex must lie in the box, and an LP must find a separating hyperplane for every pair. `proofs/independent_check.py` is a second check that does not use the LP. Certification is double precision; there are no exact rational certificates yet.
- **Notes.** `notes/experiments.md` is the lab notebook. `plan.json` is the research plan the engine follows; its reasoning is shown on the website.

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy torch
clang -O3 -ffast-math -shared -fPIC -o src/native/libkernel4d.dylib src/native/kernel4d.c
.venv/bin/python src/engine.py --ns 17-130 --minutes 1 --cpu-workers 10 --native   # then open http://localhost:8765
```

## Centre lemma (lower bound s(n) ≥ 2 for n ≥ 2)

A unit d-cube inside a d-cube of side < 2 contains the centre. This is equivalent to the following inequality: for every U ∈ O(d) and every column k,
Σ_j |U_jk|(2 − ρ_j) ≤ 1, where ρ_j is the L1 norm of row j. The numerical maximum is exactly 1 in d = 2–5 (`proofs/center_lemma_numeric.py`).
A rigorous proof for d = 4 is open. The difficulty is that the inequality is tight to second order at signed permutation matrices.

Developed by Xavier D'Mello with Claude (Anthropic).
