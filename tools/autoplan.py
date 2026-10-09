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
# Fresh-n breadth first: the gallery only shows n that have coordinates. These have never been drawn.
FRESH = [31, 32, 33, 34, 35, 36, 37, 38, 122, 123, 124, 125, 126, 127, 128, 258, 259, 260]
BREADTH = list(range(17, 31)) + [82, 83, 84, 85, 86, 101, 102, 103, 104, 105, 121, 122, 123, 124, 125, 257] + FRESH
SWEEP_NS = "31-36,122-128,258-262"          # research-core sweep: frontier, never-drawn n
MAX_MINUTES = 15                            # engine caps long tasks here; record rides only (never more)


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
    def secs(et):                                   # macOS etime: [[dd-]hh:]mm:ss
        d, _, rest = et.rpartition("-"); p = [int(x) for x in rest.split(":")]
        while len(p) < 3: p.insert(0, 0)
        return (int(d) if d else 0) * 86400 + p[0] * 3600 + p[1] * 60 + p[2]
    out = subprocess.run(["ps", "-eo", "pid=,etime=,command="], capture_output=True, text=True).stdout.splitlines()
    for line in out:
        parts = line.split(None, 2)
        if len(parts) < 3 or "src/engine.py" not in parts[2] or "results_exp_auto" in parts[2]: continue
        if re.search(r"--results results_(exp|aux)", parts[2]) and secs(parts[1]) > max_age:
            subprocess.run(["kill", "-TERM", parts[0]]); print(time.strftime("%H:%M"), "stopped stale experiment", parts[2][:90], flush=True)


TOTAL_CORES = 12


def research_workers():
    """(workers used by research engines, whether the auto sweep is running)."""
    ps = subprocess.run(["ps", "-eo", "pid=,ppid=,command="], capture_output=True, text=True).stdout.splitlines()
    rows = [l.split(None, 2) for l in ps if len(l.split(None, 2)) == 3]
    engines = {r[0]: r[2] for r in rows if "src/engine.py" in r[2] and re.search(r"--results results_(exp|aux)", r[2])}
    used = 0
    for pid in engines:
        kids = [r for r in rows if r[1] == pid and "spawn_main" in r[2]]
        used += max(0, len(kids) - 1)            # minus the Manager process
    # external research tools (e.g. polish binaries) count one core each
    used += sum(1 for r in rows if re.search(r"(^|/)(polish|polishx|pack) ", r[2] + " ") and "grep" not in r[2])
    return used, any("results_exp_auto" in c for c in engines.values())


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
        AM = float(old.get("auto_minutes", 10))
        for e in q: e["minutes"] = AM                 # auto mode: every run uses the user's max time, no exceptions
        plan = {"updated": time.strftime("%Y-%m-%d %H:%M"), "max_long": 8, "auto_minutes": AM,
                "note": ("Autonomous mode: the AI researcher is offline, so a rule-based planner is scheduling. Rules: ride n that set a "
                         "record in the last 15 min (12-min runs), grow into n+1 (8 min), and probe open breadth targets (3 min). "
                         "Skip n that failed 6 times in the last hour. Results publish automatically. "
                         f"Hot now: {hot[:3] or 'none'}; cooling off: {sorted(cold) or 'none'}."),
                "insights": old.get("insights", []), "queue": q, "auto": True}
        if old.get("auto", True):                  # manual mode (auto: false): a researcher owns plan.json; only guard the system
            json.dump(plan, open("plan.json.tmp", "w"), indent=2, ensure_ascii=False)
            os.replace("plan.json.tmp", "plan.json")
        RESEARCH_CORES = int(json.load(open("plan.json")).get("research_cores", 4))
        PROD_CORES = max(1, TOTAL_CORES - RESEARCH_CORES)
        # split changed? restart production with the new worker count (the watchdog below brings it back)
        cmd = subprocess.run(["pgrep", "-fl", "src/engine.py --ns 17-130"], capture_output=True, text=True).stdout
        m_ = re.search(r"--cpu-workers (\d+)", cmd)
        if m_ and int(m_.group(1)) != PROD_CORES:
            subprocess.run(["pkill", "-TERM", "-f", "src/engine.py --ns 17-130"]); time.sleep(4)
            print(time.strftime("%H:%M"), f"core split changed: production -> {PROD_CORES}", flush=True)
        # watchdog: the publisher must always run (pushes records to the public site within ~60 s)
        if not subprocess.run(["pgrep", "-f", "tools/publish.sh"], capture_output=True).stdout.strip():
            subprocess.Popen(["tools/publish.sh"], stdout=open("logs/publish.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
            print(time.strftime("%H:%M"), "watchdog: restarted publisher", flush=True)
        # watchdog: production must always run
        if not subprocess.run(["pgrep", "-f", "src/engine.py --ns 17-130"], capture_output=True).stdout.strip():
            p = subprocess.Popen([".venv/bin/python", "src/engine.py", "--ns", "17-130,131-400,2-16", "--minutes", "1", "--cpu-workers", str(PROD_CORES),
                                  "--gpu-workers", "0", "--native", "--cpu-batch", "48"],
                                 stdout=open("logs/dashboard.log", "w"), stderr=open("logs/engine.err", "a"), start_new_session=True)
            subprocess.Popen(["caffeinate", "-i", "-w", str(p.pid)], start_new_session=True)
            print(time.strftime("%H:%M"), "watchdog: restarted production engine", flush=True)
        stop_stale_experiments()
        used, sweeping = research_workers()
        only_sweep = sweeping and subprocess.run(["pgrep", "-f", "src/engine.py.*results_(exp|aux)"], capture_output=True, text=True).stdout.count("\n") <= 1
        if sweeping and (used > RESEARCH_CORES or (used < RESEARCH_CORES and only_sweep)):   # resize the auto sweep
            subprocess.run(["pkill", "-TERM", "-f", "results_exp_auto"]); time.sleep(3); used, sweeping = research_workers()
        free = RESEARCH_CORES - used
        if free > 0 and not sweeping:              # keep all 12 cores busy: fill the research budget with a breadth sweep
            subprocess.Popen([".venv/bin/python", "src/engine.py", "--ns", SWEEP_NS, "--minutes", str(AM if old.get("auto", True) else 5), "--cpu-workers", str(free),
                              "--gpu-workers", "0", "--native", "--port", "8889", "--results", "results_exp_auto",
                              "--label", "auto: breadth sweep (rule-based, AI offline)", "--no-dashboard"],
                             stdout=open("logs/exp_auto.out", "a"), stderr=open("logs/exp_auto.err", "a"), start_new_session=True)
            print(time.strftime("%H:%M"), f"started auto breadth sweep on {free} free research core(s)", flush=True)
        print(time.strftime("%H:%M"), "plan:", [e["n"] for e in q], "hot", hot, "cold", sorted(cold), flush=True)
    except Exception as ex:
        print(time.strftime("%H:%M"), "autoplan error:", ex, flush=True)
    # wait up to 2 min, but react within ~10 s when the Auto button flips plan.json's "auto" flag
    was = old.get("auto", True) if "old" in dir() else True
    for _ in range(12):
        time.sleep(10)
        try:
            cur = json.load(open("plan.json"))
            if (cur.get("auto", True), cur.get("auto_minutes", 10), cur.get("research_cores", 4)) != (was, old.get("auto_minutes", 10), old.get("research_cores", 4)): break
        except Exception:
            pass
