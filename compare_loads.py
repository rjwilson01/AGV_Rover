"""Does the cable load-cell sum tell you what the bar carries? (decides whether a bar sensor is worth building)

    python compare_loads.py [sim_test_log.csv] [--trim-n 30] [--thresh-n 30]

Needs a run with `--set loadcell.bar_enabled=true` (yaw-tube build). For each hitch it fits
    bar_axial = a * (cable_sum - baseline) + b
where the baseline is the resting cable sum (first 0.5 s), then reports the share `a`, the residual error in N, and how well
the cable sum detects compression (bar force below -thresh). Verdict: if the residual is well inside the control trim target
(--trim-n) the bar sensor adds nothing; if not, build it.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np


def load(path):
    lines = [l for l in open(path, encoding="utf-8") if not l.startswith("#")]
    rows = list(csv.reader(lines))
    head = rows[0]
    data = [r for r in rows[1:] if len(r) == len(head)]
    return head, data


def column(head, data, name):
    i = head.index(name)
    return np.array([float(r[i]) if r[i] != "" else np.nan for r in data])


def compare(t, cable_sum, bar, baseline_s=0.5, thresh_n=30.0, trim_n=30.0):
    """Returns a dict of statistics for one hitch (pure function, tested)."""
    m = np.isfinite(cable_sum) & np.isfinite(bar)
    t, cs, bar = t[m], cable_sum[m], bar[m]
    rest = t <= (t[0] + baseline_s)
    base = float(np.mean(cs[rest])) if rest.any() else float(cs[0])
    proxy = cs - base
    A = np.vstack([proxy, np.ones_like(proxy)]).T
    (a, b), *_ = np.linalg.lstsq(A, bar, rcond=None)
    fit = a * proxy + b
    resid = bar - fit
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((bar - bar.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms = float(np.sqrt(np.mean(resid ** 2)))
    true_c = bar < -thresh_n                          # truly in compression (bar pushed)
    est_c = fit < -thresh_n                           # compression as seen through the cables
    tp = int(np.sum(true_c & est_c))
    out = dict(slope=float(a), offset=float(b), r2=float(r2), rms_n=rms, baseline_n=base, n=int(len(bar)),
               compression_samples=int(true_c.sum()), detected=tp,
               recall=(tp / true_c.sum()) if true_c.any() else None,
               false_alarms=int(np.sum(est_c & ~true_c)))
    if a < 0:
        verdict = "slope is NEGATIVE: flip loadcell.bar_force_sign (a pull should read positive) and rerun"
    elif rms <= 0.5 * trim_n and (out["recall"] is None or out["recall"] >= 0.9):
        verdict = f"cable sum tracks the bar to {rms:.1f} N RMS (< half of the {trim_n:g} N trim target): bar sensor NOT needed"
    else:
        verdict = f"residual {rms:.1f} N RMS vs {trim_n:g} N trim target (or compression missed): a bar sensor would add information"
    out["verdict"] = verdict
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="sim_test_log.csv")
    ap.add_argument("--trim-n", type=float, default=30.0)
    ap.add_argument("--thresh-n", type=float, default=30.0)
    a = ap.parse_args(argv)
    head, data = load(a.csv)
    t = column(head, data, "timestamp") if "timestamp" in head else column(head, data, head[0])
    hitches = sorted({int(c.split("/")[0][5:]) for c in head if c.startswith("joint") and c.endswith("/bar_axial_N")})
    if not hitches:
        print("no joint*/bar_axial_N columns: rerun with  --set loadcell.bar_enabled=true  (yaw-tube build)")
        return 1
    for j in hitches:
        r = compare(t, column(head, data, f"joint{j}/cable_sum_N"), column(head, data, f"joint{j}/bar_axial_N"),
                    thresh_n=a.thresh_n, trim_n=a.trim_n)
        print(f"hitch {j}: share a={r['slope']:+.2f}  offset {r['offset']:+.1f} N  R2={r['r2']:.3f}  RMS error {r['rms_n']:.1f} N  "
              f"(baseline cable sum {r['baseline_n']:.0f} N, {r['n']} samples)")
        print(f"         compression: {r['compression_samples']} samples, detected {r['detected']}, false alarms {r['false_alarms']}")
        print("         ->", r["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
