# Theorist notes (2026-10-09, ~14:15-15:00)

## T1. Conference-matrix centre cube: s(k^4+1) <= k + 1/sqrt3  -- CERTIFIED, NEW RECORDS
- **Hypothesis.** Put 16 axis-aligned blocks around a point where gaps of width g cross on all 4 axes. A centre
  cube with rotation R misses block sigma iff some w in the simplex has (1/2)||R^T(sigma o w)||_1 <= g/2 (separating
  hyperplane with nonnegative normal). Choosing w = 1/3 on three coordinates, the bound is g = 1/sqrt3 provided
  every sigma has a 3-coordinate signed diagonal equal to +-(a cube axis). That holds iff the cube axes are
  (0,+-1,+-1,+-1)/sqrt3 patterns: the columns of the order-4 conference matrix C = [[0,1,1,1],[1,0,1,-1],[1,-1,0,1],[1,1,-1,0]]^T.
  The 8 vectors +-c_m have one zero each, so each covers 2 orthants, and together they cover all 16.
- **Numerics.** I minimised max_sigma min_w (LP) over SO(4) with Nelder-Mead. The best I found was g = 0.57751, i.e. 1/sqrt3 to 4 digits.
  In 3D the same optimisation gives g = 1/sqrt2, so this is a genuinely 4D effect.
- **Result.** Built with src/theory_seeds.py `k4_plus_one(k)` and certified with geometry.certify:
  - n=17: s = 2 + 1/sqrt3 = **2.577350** (was 2.578147 from search). Search had essentially found this structure.
  - n=82: s = 3 + 1/sqrt3 = **3.577350** (was 3.707107). Production imported it at 14:42.
  - n=257: s = 4.577350 (certified, file results_exp_theory_A/best/d4_n257.json; outside production range).
  - General: n = k^4+1 at k + 1/sqrt3. Putting more gaps (p gaps per axis -> k^4+p^4 at k+p/sqrt3) never beats the grid.
- Conjecture: s(17) = 2 + 1/sqrt3 (the 4D analogue of s(5) = 2 + 1/sqrt2 in 2D).

## T2. 3D lift of a "9-cube centre trick" for n=18 -- NEGATIVE
- **Hypothesis.** If the 3D centre gap were < 1/sqrt2, then (8+1)-cube 3D x [0,2] would give n=18 below 2.7071.
- **Result.** The 3D optimum of the same functional is exactly 1/sqrt2. No gain.

## T3. Growing from the new structural seeds (engine, 1 core each, grow/best)
- A: `--ns 18 --strategies grow,best --results results_exp_theory_A`, seeded by the new 17.
- B: `--ns 83 --strategies grow --results results_exp_theory_B`, seeded by the new 82.
- Result: see the final report. Nothing below 2.7071 (n=18) or 3.7071 (n=83) in about 10 minutes.

## T4. Seed generators (src/theory_seeds.py; nothing in engine.py is changed)
- `cross_cell_seed(n, s)`: 5x5-product frame plus (n-25) "cross cells" (x_i = x_j = mid, with i in {1,2} and j in {3,4}, turned 45 deg
  in plane (i,j)). I rebuilt it from the verbal description only. The seed is rough and infeasible, so the engine has to relax it.
  Note: at s ~ 2.94 a raw cross cell overlaps the frame's one-plane 45-degree cubes (that overlap only vanishes for s >= 3.414),
  so the frame has to be "opened" (cubes slide). The seed needs relaxation, not just inflation.
- `hybrid17_seed(n, s)`: 16 corners + conference centre + (n-17) of the 8 one-plane 45-degree cells, for n = 18..24.
- `product_frame_seed(A, B, extra, s)`: generalises the cross-cell idea to any 2D x 2D product (for example 10x10 -> 101+).
- To use them, the engine needs a "seed" strategy: init from a generator instead of from board[n]. That needs a small engine.py change, which I did not make.
