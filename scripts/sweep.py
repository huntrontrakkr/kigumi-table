"""Build + check many random designs; report crashes and check outcomes.
    uv run python scripts/sweep.py N [start_seed]"""
import sys
import time
import traceback
from collections import Counter

from kigumi.build import build
from kigumi.checks import run_checks
from kigumi.params import sample

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
stats = Counter()
fails = Counter()
crashes = []
times = []
for s in range(s0, s0 + N):
    p = sample(s)
    t = time.time()
    try:
        d = build(p)
        res = run_checks(d)
    except Exception as e:
        crashes.append((s, repr(e)[:200], traceback.format_exc().splitlines()[-3:]))
        continue
    times.append(time.time() - t)
    worst = "pass"
    for r in res:
        if r["status"] == "fail":
            fails[r["name"]] += 1
            worst = "fail"
        elif r["status"] == "warn" and worst == "pass":
            worst = "warn"
    stats[worst] += 1
    nb = len(d.layout.beams)
    bad = ",".join(r["name"].split(" (")[0] for r in res if r["status"] == "fail")
    print(f"seed {s}: {p.grammar:6s} {p.top_shape:8s} legs={len(d.layout.legs)} tiers={len(d.layout.z_tiers)} beams={nb:3d} "
          f"{worst:4s} {time.time() - t:.2f}s  {bad + ' | ' if bad else ''}{'; '.join(d.layout.notes)[:90]}", flush=True)
print("\nverdicts", dict(stats))
print("failing checks", dict(fails))
print("crashes", len(crashes))
for c in crashes:
    print(c)
if times:
    print(f"time per design: mean {sum(times) / len(times):.2f}s max {max(times):.2f}s")
