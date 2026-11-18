"""Curvature-limited path planner.

Takes the rough waypoint polyline from the config and rounds every corner with a circular arc, giving a smooth,
densely sampled path whose curvature stays inside what the articulated chain can follow.

Why: a chain of two-wheel trailers can only follow a turn of radius R if each hitch can articulate enough. For a
hitch with rear bar L_r (lead side) and front bar L_f (follower side), steady-state hitch angle on a circle of lead-axle
radius R is  psi = atan(L_r/R) + atan(L_f/R_f),  R_f^2 = R^2 + L_r^2 - L_f^2.  Setting psi to the usable hitch range
gives the tightest radius the chain can track. Tighter than that and the hitches hit their stops and the chain is
dragged around by the cables (what the U-turn log showed: lead pinned at kappa 0.55, hitch beyond 60 deg).
"""
import math
from dataclasses import dataclass, field

import numpy as np


def hitch_angle(R: float, L_r: float, L_f: float) -> float:
    """Steady-state hitch yaw (rad) when the lead axle follows a circle of radius R."""
    rf_sq = R * R + L_r * L_r - L_f * L_f
    if rf_sq <= 0.0:
        return math.pi / 2
    return math.atan(L_r / R) + math.atan(L_f / math.sqrt(rf_sq))


def hitch_limited_curvature(L_r: float, L_f: float, psi_max_rad: float) -> float:
    """Largest curvature (1/m) at which the hitch angle stays <= psi_max_rad (bisection on radius)."""
    lo, hi = 0.05, 1000.0                       # radius bounds; hitch_angle decreases as R grows
    if hitch_angle(hi, L_r, L_f) > psi_max_rad:
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if hitch_angle(mid, L_r, L_f) > psi_max_rad:
            lo = mid
        else:
            hi = mid
    return 1.0 / hi


@dataclass
class PlanResult:
    points: np.ndarray
    kappa_limit: float
    corner_radii: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def min_radius(self) -> float:
        return min(self.corner_radii) if self.corner_radii else math.inf


def plan_path(waypoints, kappa_limit: float, spacing_m: float = 0.05, radius_cap_m: float = 8.0, runout_m: float = 0.0) -> PlanResult:
    wp = np.asarray(waypoints, dtype=float)
    res = PlanResult(points=wp.copy(), kappa_limit=kappa_limit)
    if len(wp) < 3:
        return res
    r_min = 1.0 / kappa_limit if kappa_limit > 0 else math.inf
    seg = wp[1:] - wp[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    unit = seg / np.maximum(seg_len[:, None], 1e-9)

    pts = [wp[0]]
    for i in range(1, len(wp) - 1):
        u1, u2 = unit[i - 1], unit[i]
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        delta = math.atan2(abs(cross), float(np.dot(u1, u2)))        # turning angle at this corner
        if delta < math.radians(0.5):
            pts.append(wp[i])
            continue
        # Tangent length available: share a segment with its neighbouring corner, but a segment that ends at a
        # path endpoint can be used in full.
        a1 = seg_len[i - 1] if i - 1 == 0 else 0.5 * seg_len[i - 1]
        a2 = seg_len[i] if i == len(wp) - 2 else 0.5 * seg_len[i]
        t_avail = min(a1, a2)
        t_wanted = min(radius_cap_m, 1e6) * math.tan(delta / 2)
        T = min(t_avail, t_wanted)
        R = T / math.tan(delta / 2)
        res.corner_radii.append(R)
        if R < r_min - 1e-9:
            res.warnings.append(
                f"corner {i} at ({wp[i][0]:.1f}, {wp[i][1]:.1f}): turn {math.degrees(delta):.0f} deg fits radius "
                f"{R:.2f} m but the chain needs >= {r_min:.2f} m; space the waypoints further apart"
            )
        p1 = wp[i] - u1 * T
        side = 1.0 if cross > 0 else -1.0                              # +1 = left turn
        centre = p1 + side * np.array([-u1[1], u1[0]]) * R
        h1 = math.atan2(u1[1], u1[0])
        n = max(2, int(math.ceil(R * delta / spacing_m)))
        for k in range(n + 1):
            h = h1 + side * delta * k / n
            # point on circle at heading h
            pts.append(centre + R * np.array([math.sin(h) * side, -math.cos(h) * side]))
    pts.append(wp[-1])
    if runout_m > 0.0:                            # straight run-out so the lookahead never collapses onto the goal point
        pts.append(wp[-1] + unit[-1] * runout_m)
    pts = np.asarray(pts)

    # Resample at uniform spacing along the arc length
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(d) > 1e-9])
    pts, d = pts[keep], d[keep]
    s = np.arange(0.0, d[-1], spacing_m)
    s = np.append(s, d[-1])
    res.points = np.column_stack([np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])])
    return res


def path_curvature(points: np.ndarray) -> np.ndarray:
    """Signed curvature at each interior sample (left positive), from heading change over arc length."""
    d = np.diff(points, axis=0)
    ds = np.linalg.norm(d, axis=1)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.diff(h) / np.maximum(0.5 * (ds[1:] + ds[:-1]), 1e-9)


def chain_curvature_limit(cfg, margin: float = 0.85) -> float:
    """Hitch-limited curvature for this config. Variable bars can shorten to their minimum in tight turns."""
    var = cfg.rear_bar_range_m[1] > cfg.rear_bar_range_m[0] or cfg.front_bar_range_m[1] > cfg.front_bar_range_m[0]
    if var:
        L_r, L_f = cfg.rear_bar_range_m[0], cfg.front_bar_range_m[0]
    else:
        L_r, L_f = cfg.nominal_rear_bar_m, cfg.nominal_front_bar_m
    psi = math.radians(min(abs(cfg.yaw_angle_range_deg[0]), abs(cfg.yaw_angle_range_deg[1]))) * margin
    return hitch_limited_curvature(L_r, L_f, psi)
