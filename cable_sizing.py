"""Cable-actuator sizing numbers from a run log (for design.toml and the motor / drum choice).

    python cable_sizing.py [sim_test_log.csv] [--drum-dia-mm 60] [--seg SEGMENT]

Run the log from `t_walk_cable_load` (walking pace, bar load logged). Per hitch and cable corner it reports
  - peak and mean tension (N)            -> cable, strain gauge, motor torque
  - cable travel, max - min length (mm)  -> drum length / turns
  - peak cable speed (m/s)               -> motor speed; with --drum-dia-mm also drum rpm and torque per 100 N
and, per hitch, the bar axial load (min = compression, max = pull) when the run logged it.
Cable speed is the time derivative of the measured cable length, smoothed over ~0.1 s.
"""
from __future__ import annotations

import argparse

import numpy as np

from compare_loads import column, load

CORNERS = ("tl", "tr", "bl", "br")


def _speed(t: np.ndarray, length: np.ndarray, win_s: float = 0.1) -> np.ndarray:
    """|d length / dt| with a moving average of about win_s."""
    m = np.isfinite(t) & np.isfinite(length)
    t, length = t[m], length[m]
    if len(t) < 3:
        return np.zeros(0)
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(win_s / dt))) if dt > 0 else 1
    if k > 1:       # edge-padded moving average (zero padding would fake a huge speed at both ends)
        pad = np.pad(length, (k // 2, k - 1 - k // 2), mode="edge")
        ls = np.convolve(pad, np.ones(k) / k, mode="valid")
    else:
        ls = length
    return np.abs(np.gradient(ls, t))


def summarise(head, data, drum_dia_mm: float | None = None) -> dict:
    t = column(head, data, "timestamp")
    out: dict = {"hitches": {}}
    j = 1
    while f"joint{j}/yaw_meas_deg" in head:
        h: dict = {"cables": {}}
        for c in CORNERS:
            ten_name, len_name = f"joint{j}/tension_{c}_N", f"joint{j}/cable_{c}_meas_m"
            if ten_name not in head or len_name not in head:
                continue
            ten, ln = column(head, data, ten_name), column(head, data, len_name)
            v = _speed(t, ln)
            d = {"tension_peak_N": float(np.nanmax(ten)), "tension_mean_N": float(np.nanmean(ten)),
                 "travel_mm": float((np.nanmax(ln) - np.nanmin(ln)) * 1000.0),
                 "speed_peak_mps": float(v.max()) if v.size else float("nan")}
            if drum_dia_mm:
                circ = np.pi * drum_dia_mm / 1000.0
                d["drum_rpm_peak"] = d["speed_peak_mps"] / circ * 60.0
                d["drum_turns"] = d["travel_mm"] / 1000.0 / circ
                d["drum_torque_peak_Nm"] = d["tension_peak_N"] * drum_dia_mm / 2000.0
            h["cables"][c] = d
        for name in ("bar_axial_true_N", "bar_axial_N"):
            if f"joint{j}/{name}" in head:
                b = column(head, data, f"joint{j}/{name}")
                h[name] = {"min_N": float(np.nanmin(b)), "max_N": float(np.nanmax(b))}
        out["hitches"][j] = h
        j += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="sim_test_log.csv")
    ap.add_argument("--drum-dia-mm", type=float, default=None)
    ap.add_argument("--seg", default=None, help="only rows whose test/name contains this text")
    a = ap.parse_args()
    head, data = load(a.log)
    if a.seg and "test/name" in head:
        i = head.index("test/name")
        data = [r for r in data if a.seg in r[i]]
    if not data:
        raise SystemExit("no rows to summarise")
    s = summarise(head, data, a.drum_dia_mm)
    worst = {"tension_peak_N": 0.0, "travel_mm": 0.0, "speed_peak_mps": 0.0}
    for j, h in s["hitches"].items():
        print(f"hitch {j}")
        for c, d in h["cables"].items():
            extra = (f"  drum {d['drum_rpm_peak']:.0f} rpm, {d['drum_turns']:.2f} turns, {d['drum_torque_peak_Nm']:.2f} N.m"
                     if a.drum_dia_mm else "")
            print(f"  {c}: peak {d['tension_peak_N']:.1f} N (mean {d['tension_mean_N']:.1f}), travel {d['travel_mm']:.0f} mm, "
                  f"speed {d['speed_peak_mps']:.3f} m/s{extra}")
            for k in worst:
                worst[k] = max(worst[k], d[k])
        for name in ("bar_axial_true_N", "bar_axial_N"):
            if name in h:
                print(f"  {name}: {h[name]['min_N']:.1f} .. {h[name]['max_N']:.1f} N")
    print(f"worst case over all cables: tension {worst['tension_peak_N']:.1f} N, travel {worst['travel_mm']:.0f} mm, "
          f"speed {worst['speed_peak_mps']:.3f} m/s   (-> design.toml [cable] / [[active_end]])")
    if not any("bar_axial_true_N" in h or "bar_axial_N" in h for h in s["hitches"].values()):
        print("note: no bar load channel in this log; run t_walk_cable_load (sets loadcell.bar_enabled = true)")


if __name__ == "__main__":
    main()
