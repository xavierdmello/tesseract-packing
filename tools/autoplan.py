"""Autonomous planner: keeps plan.json sensible while no AI researcher is available.

    nohup .venv/bin/python tools/autoplan.py >> logs/autoplan.log 2>&1 &

Every 2 minutes it reads results/events.log and results/live.json and writes plan.json with simple rules:
  1. HOT: n that set a record in the last 15 min      -> long task (12 min), max 3 long slots
  2. NEXT: n+1 of a hot n (grow from the new record)   -> 8 min
  3. OPEN: breadth targets not yet searched recently   -> 3 min probes (round robin)
  4. COLD: n searched >= 6 times in the last hour with no record -> skipped for an hour
It also keeps the 4 research cores busy: if no experiment engine is running, it starts a 4-worker breadth sweep
(results_exp_auto) whose certified records production imports automatically.
"""
import json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BREADTH = list(range(17, 31)) + [82, 83, 84, 85, 86, 101, 102, 103, 104, 105, 121, 122, 123, 124, 125, 257]
SWEEP_NS = "102-110,122-128,31-36"


def parse_events(since):
    recs, tasks = {}, {}
    today = time.strftime("%Y-%m-%d ")
    for line in open("results/events.log", errors="ignore"):
        m = re.match(r"(\d\d:\d\d:\d\d) n=(\d+): ([\d.]+) -> ([\d.]+)", line)
        m2 = re.match(r"(\d\d:\d\d:\d\d) worker \d+: n=(\d+) done, (IMPROVED|no improvement)", line)
        mm = m or m2
        if not mm: continue
        t = time.mktime(time.strptime(today + mm.group(1), "%Y-%m-%d %H:%M:%S"))
        if t > time.time() + 60: t -= 86400          # yesterday's lines
        if t < since: continue
        n = int(mm.group(2))
        if m: recs[n] = max(recs.get(n, 0), t)
        else: tasks.setdefault(n, []).append((t, mm.group(3) == "IMPROVED"))
    return recs, tasks


def stop_stale_experiments(max_age=1800):
    """Experiment engines loop forever; stop ones (other than the auto sweep) older than max_age seconds."""
    out = subprocess.run(["ps", "-eo", "pid,etimes,command"], capture_output=True, text=True).stdout.splitlines()
    for line in out[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3 or "src/engine.py" not in parts[2] or "results_exp_auto" in parts[2]: continue
        if re.search(r"--results results_(exp|aux)", parts[2]) and int(parts[1]) > max_age:
            subprocess.run(["kill", "-TERM", parts[0]]); print(time.strftime("%H:%M"), "stopped stale experiment", parts[2][:90], flush=True)


def research_running():
    out = subprocess.run(["pgrep", "-f", "results_(exp|aux)"], capture_output=True, text=True).stdout.split()
    return len(out) > 0


rr = 0
while True:
    try:
        now = time.time()
        recs, tasks = parse_events(now - 3600)
        hot = sorted([n for n, t in recs.items() if now - t < 900], key=lambda n: -recs[n])
        cold = {n for n, ts in tasks.items() if len(ts) >= 6 and not any(ok for _, ok in ts) and n not in recs}
        q = []
        for n in hot[:3]:
            q.append({"n": n, "minutes": 12, "why": f"Hot: set a record {int((now - recs[n]) / 60)} min ago. Long run while it's still falling."})
        for n in hot[:3]:
            if n + 1 not in hot and n + 1 not in cold and n + 1 <= 257:
                q.append({"n": n + 1, "minutes": 8, "why": f"Grow from the new n = {n} record."})
        opens = [n for n in BREADTH if n not in cold and all(e["n"] != n for e in q)]
        rr = (rr + 3) % max(1, len(opens))
        for n in (opens[rr:] + opens[:rr])[:10]:
            q.append({"n": n, "minutes": 3, "why": "Breadth probe (round robin over open targets)."})
        old = json.load(open("plan.json")) if os.path.exists("plan.json") else {}
        plan = {"updated": time.strftime("%Y-%m-%d %H:%M"), "max_long": 3,
                "note": ("Autonomous mode: the AI researcher is offline, so a rule-based planner is scheduling. Rules: ride n that set a "
                         "record in the last 15 min (12-min runs), grow into n+1 (8 min), and probe open breadth targets (3 min). "
                         "Skip n that failed 6 times in the last hour. Results publish automatically. "
                         f"Hot now: {hot[:3] or 'none'}; cooling off: {sorted(cold) or 'none'}."),
                "insights": old.get("insights", []), "queue": q, "auto": True}
        json.dump(plan, open("plan.json.tmp", "w"), indent=2, ensure_ascii=False)
        os.replace("plan.json.tmp", "plan.json")
        stop_stale_experiments()
        if not research_running():
            subprocess.Popen([".venv/bin/python", "src/engine.py", "--ns", SWEEP_NS, "--minutes", "5", "--cpu-workers", "4",
                              "--gpu-workers", "0", "--native", "--port", "8889", "--results", "results_exp_auto",
                              "--label", "auto: breadth sweep (rule-based, AI offline)", "--no-dashboard"],
                             stdout=open("logs/exp_auto.out", "a"), stderr=open("logs/exp_auto.err", "a"), start_new_session=True)
            print(time.strftime("%H:%M"), "started auto breadth sweep on research cores", flush=True)
        print(time.strftime("%H:%M"), "plan:", [e["n"] for e in q], "hot", hot, "cold", sorted(cold), flush=True)
    except Exception as ex:
        print(time.strftime("%H:%M"), "autoplan error:", ex, flush=True)
    time.sleep(120)
