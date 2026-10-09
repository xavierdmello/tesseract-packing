# ZCode research log — Head Researcher handoff (2026-10-09, 15:00–17:00 shift)

Written by ZCode (GLM-5.3-Flash), acting Head Researcher while Claude was away (back ~17:00).
This is the protocol + findings doc for whoever holds the Head Researcher seat next.
The ops manual lives in this doc; the lab notebook is `notes/experiments.md`; theory in `notes/theorist.md`.

## 1. What the Head Researcher actually does

The machine has 12 cores, split **8 production + 4 experimental**:

- **Production (8 cores):** one `src/engine.py` farm process (`--ns 17-130,257 --minutes 1 --cpu-workers 8
  --native --cpu-batch 48`), writing `results/`. It follows `plan.json` **only** — the queue, the minutes per n,
  the reasoning. You write `plan.json` (set `"auto": false` first so autoplan doesn't fight you).
- **Experimental (4 cores):** whatever experiments you judge most valuable, one core each
  (`src/engine.py --results results_exp_<name> --port 87xx --no-dashboard` or external binaries).
  Certified records from any `results_exp*/best/d4_nN.json` are **auto-imported by production every 30 s**.
- **Publisher (`tools/publish.sh`, always running):** pushes the public site within 60 s of any record change,
  commits `research` every 5 min. Never run it twice; it force-pushes `site` as a single amend commit.
- **`tools/autoplan.py` = "auto mode":** the rule-based planner that schedules when no AI researcher is seated.
  Auto mode ON = autoplan running and `plan.json.auto = true`. Auto mode OFF = kill autoplan, write plans yourself.

Head Researcher duties, in order of value:
1. **Think** — derive structures, predict where the next opening is, write seed generators. Every 4D record so far
   came from structure, never from raw search. Run theorist subagents; have them propose, you decide.
2. **Allocate the 4 experimental cores** — reproduce/verify rival results, A/B candidate algorithms, attack new n.
3. **Ride hot records** — when an n is falling (multiple records within minutes), put a long run on it (cap 20 min,
   30 min if truly exceptional) before the vein closes.
4. **Keep breadth** — always keep queue slots for **fresh n** (never-drawn, still grid) so the gallery grows:
   the gallery only shows n with coordinates.
5. **Promote winners** — if an experimental method beats production's method, wire it into `src/engine.py`
   (see §3 for the two promotions that happened today) and say so in `plan.json.insights`.

## 2. State at end of this shift (2026-10-09 ~16:00)

| n | s | how |
|---|---|---|
| 17 | 2.577350 | conference construction (2+1/√3), conjectured optimal |
| 18–25 | 2.707107 | 5×5 product |
| 26–29 | 2.942809–2.942834 | **hockyy's packings, imported + independently certified** (≈2+2√2/3 conjecture) |
| 30 | 2.989568 | hockyy, imported + certified |
| 31–80 | 3.0 | grid — **open target: sub-3.0 at 31–33 may be one cross cell away** |
| 81 | 3.0 | optimal by volume |
| 82 | 3.577350 | conference construction |
| 83–100 | 3.707115 | 10×10 product |
| 101 | **3.717655** | AL-polished twice (was 3.7737 pre-polish) |
| 102 | 3.8435 | growing; falling all afternoon |
| 103–121 | 3.879404 | 11×11 product |
| 122+ | 4.0 | grid; X5-style frame+cross-cells is the only known route |

Rival: [hockyy/tesseract-packing](https://github.com/hockyy/tesseract-packing) — we **match** their published
26–30 (imported, certified by our LP checker, credited to them). We have not **beaten** them anywhere yet.

## 3. What this shift changed (for future researchers)

1. **E9 — hockyy pipeline reproduced from scratch.** Built their `pack/polish/polishx` from source (needs the
   `bits/stdc++.h` shim in `/tmp/hockyy_inc` and `SDKROOT` pointed at Xcode's SDK — Apple CLT's new tbd files break
   the CLT linker; see §6). Their pipeline from scratch: n=26 within 3e-6 of their published value in 3 min;
   n=27/28 within ~7e-4; n=30 from scratch stalls at 3.0000000001 (their exact seed selection matters).
   Verdict: their published files remain the best known; their *method* is the best known.
2. **Matched 26–30.** Imported all 5 published packings (R row-major, no transpose — verify with `geometry.certify`),
   certified, production imported them at 15:42, site/README updated with credit.
3. **P1 — promoted their AL+L-BFGS polisher into main.** `src/polish_bridge.py` wraps their `polish` binary;
   `engine.py accept()` now polishes **every record candidate before certification** (all 8 cores, all n,
   and every experiment import). First payoff within 2 minutes: **n=101 3.7737 → 3.7177** (polished twice:
   once manually, once again on import). Rules: keep polished result only if strictly better AND certified;
   any bridge failure falls back to the raw candidate. NOTE: the binary lives in `/tmp/hockyy_tp` — rebuild
   recipe in §6; the bridge degrades gracefully if it's missing.
4. **Site: experiments table.** `results/experiments.json` (id, name, n, best, target, status, note) is rendered
   in a new "Experiments — 4 research cores" section (local page polls it live; the static build bakes it into
   `data.json`). Keep it updated when you start/finish experiments — that's how the public sees the research loop.
5. **X2 (running at shift end):** grow n=31–33 from the imported n=30, 4 cores, 30 min (`results_exp_hk2`).
   At 15:55: task bests 3.00001–3.00006, still relaxing. If it lands below 3.0, that's a first.
6. **Auto mode upgraded:** autoplan now (a) caps record rides at 20 min, (b) always round-robins fresh-n breadth
   probes (31–36, 122–128, 258–262), (c) its research-core sweep targets the frontier, (d) benefits from the
   polish promotion automatically (it only restarts `engine.py`).
7. **Auto button on the site (local only):** POST `/auto` on the engine's port toggles autoplan
   (start/stop + `plan.json.auto`). The public build hides it automatically (it sits in the hidden live section).

## 4. The theory frontier (what to think about next)

These were posed to the theory agent this shift; it was stopped early, so they're open:

1. **The opening mechanism.** Derive the exact gap width g(s) available to a cross cell (45° in a mixed plane
   (i,j)) on the opened 25-frame, as a function of s. Confirm the threshold where k cells fit. Predict exact
   s for n=31,32,33 in the frame class — this decides whether X2 can succeed and where the wall is.
2. **Conference-hybrid frame (possible goldmine).** hockyy's centre cube is 45° in (0,1)×(2,3). T1's conference
   rotation needs only 1/√3 diagonal clearance. Replace the centre cube (and/or the 8 one-plane cubes) with
   conference-rotated cubes: how many extra cubes fit below s=3.0? Does it beat 5 cross cells (n=30)?
   Build concrete seeds (`{c,R}` json) so the engine can eat them.
3. **k⁴+m family.** s(k⁴+1) ≤ k+1/√3 is certified (17, 82, 257). What is the best k⁴+2? Two conference cubes
   need disjoint simplex directions — count how many disjoint w's the orthant cover supports. Even
   s(18) ≤ 2+c with c<0.7071 would beat the 5×5 product at 18–25. Nobody has moved 18–25 all day.
4. **Fresh-n gallery seeds.** Cheap closed-form constructions for never-drawn n: 122–130 (10×10 frame +
   cross cells at s≈3.96, `theory_seeds.product_frame_seed` + `results_aux2/best/d2_n10.json`), 258–263
   (4⁴+m), 37–40 (3⁴ frame opened). Seed files → drop in a `results_exp_*/best/` dir → production imports.
5. **Lower bounds.** Centre lemma gives s≥2 (numerically verified d=2–5, proof open in d=4). For n=26 the gap
   2.258 (volume) → 2.9428 (best) is enormous. A class-restricted bound ("in the opened-frame+cross-cell class,
   at most m cubes fit below s") via LP infeasibility enumeration over the 16 cross cells × symmetry orbits
   would already be site-worthy if stated honestly as class-restricted.
6. **Marginal-cost curve (verified this shift, theorist round 1).** 25→26 costs +0.2357 (frame opening), cells
   2–5 cost ~1e-5, 29→30 costs +0.0468 (second opening). On the 10×10 frame the first cell costs only +0.0666
   (our n=101). If a 6th cell's cost is another ~0.047, n=31 lands ≈3.037 (X2 fails); if ~1e-3, it lands ≈2.990
   (X2 wins). Computing this from the geometry beats running searches blind.

## 5. Rules of the road (learned the hard way today)

- **Never let uncertified coordinates into `results/`** — the engine's `certify_inflate` + LP check is the gate;
  experiment imports go through it. Rough seeds live in `results_exp_*/best/` marked `"certified": false` and
  are only used as starting points.
- **Experiments must opt out of plan.json** (`--results results_exp_*` does this) and use their own port (87xx).
- One experiment engine per results dir; experiment engines older than 30 min get killed by autoplan
  (`stop_stale_experiments`) — for longer runs, say so in the docstring and exclude the dir.
- The production engine rebuilds its board from `results/best` on restart; restarting is safe and is how you
  load engine.py changes. Workers die with the parent (getppid check).
- `plan.json` is read on every task assignment; bump `"updated"` to force a reload. Engine caps: `max_long`
  long tasks at once; tasks longer than `--minutes 1` count as long. Minutes per entry are capped at 15 by
  `mins_of` — for 20–30 min rides raise that cap in `engine.py next_task` or run them as experiments.
- Credit rivals by name in `source` fields and the README. We re-certified hockyy's work before adopting it;
  do the same for anything external.
- The site's "Found" captions show the date only; keep sensitive/verbose notes in `notes/`, not in result files.

## 6. Build recipe for hockyy's binaries (macOS, Apple clang)

```bash
git clone --depth 5 https://github.com/hockyy/tesseract-packing /tmp/hockyy_tp
mkdir -p /tmp/hockyy_inc/bits   # Apple clang has no bits/stdc++.h
# write a bits/stdc++.h that includes the used std headers (algorithm, atomic, cmath, ...) + unistd.h
cd /tmp/hockyy_tp
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
for f in pack polish polishx; do
  clang++ -O3 -ffast-math -pthread -I/tmp/hockyy_inc src/$f.cpp -o $f
done
# pipeline: python3 src/tenum.py N seeds.json S0 jit reps seed   (symmetry-orbit cross-cell seeds)
#           BH=200 THREADS=1 ./polish seeds.json out.json <topk>   (AL+L-BFGS polish + basin hopping)
#           ./pack d n starts soft_steps seed out.json [threads]   (soft-to-rigid random starts)
# JSON gotcha: their C++ parser greps "centers":[ with no space — dump with separators=(",",":").
```

## 7. If you are the next AI researcher (quick start)

1. `cat plan.json` — the note + queue is the current strategy. If `auto:true`, either take over
   (kill autoplan, set auto:false) or leave it and just audit.
2. `tail -50 results/events.log` — what's falling right now.
3. `cat results/experiments.json` — what's running/finished on the 4 research cores.
4. Write your plan.json (auto:false), keep 3–5 record rides + ≥4 fresh-n probes, respect the 20-min cap.
5. Start experiments from §4 (highest value: the opening-mechanism derivation and the conference hybrid).
6. Before 17:00 (or whenever your shift ends): update `notes/experiments.md`, `docs/ZCODE_RESEARCH.md`,
   restart autoplan (auto mode ON), commit, and let publish.sh push.

*Shift log: took over 15:30 → autoplan stopped 15:31 → E9 built+run 15:33–15:47 → imports landed 15:42 →
experiments table on site 15:45 → X2 launched 15:47 → AL polish promoted 15:55 → n=101 −0.056 by 15:57 →
theory agent stopped early on user request 16:00 → this doc written 16:00.*
