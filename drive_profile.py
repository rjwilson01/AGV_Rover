"""Open-loop drive profile for impulse / step-response testing.

Spec: comma-separated windows  start_s:end_s:speed_mps:curvature_per_m   (curvature > 0 = left turn)
Outside every window the lead is commanded to stand still (0 m/s, 0 curvature).

  Straight, then a 0.5 s curvature impulse, then straight:   0:3:1.0:0,3:3.5:1.0:0.25,3.5:12:1.0:0
  Speed step from 0.5 to 1.0 m/s on a straight:              0:4:0.5:0,4:12:1.0:0
  Hold a constant 0.15 /m left turn:                         0:15:1.0:0.15
"""


def parse_drive(spec: str):
    windows = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        f = part.split(":")
        if len(f) != 4:
            raise ValueError(f"bad drive window '{part}': want start_s:end_s:speed_mps:curvature_per_m")
        t0, t1, v, k = (float(x) for x in f)
        if t1 <= t0:
            raise ValueError(f"bad drive window '{part}': end must be after start")
        windows.append((t0, t1, v, k))
    if not windows:
        raise ValueError("empty drive spec")
    windows.sort()

    def profile(t: float):
        for t0, t1, v, k in windows:
            if t0 <= t < t1:
                return v, k
        return 0.0, 0.0

    profile.windows = windows
    return profile
