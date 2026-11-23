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


# --- Three-part drawbar (bar tube pivot - centre part with one vertical yaw pin - bar tube pivot) ------------------
# Plan view, left turn, lead trailer centre at the origin heading +x, turn centre O = (0, R). Hinges in series:
#   psi_r  rear tube pivot on the lead trailer (cable-limited, about +-90 deg)
#   theta  vertical clevis pin of the centre part (Balljoint part.fcmacro: pin 158 mm from the pitch pin)
#   psi_f  front tube pivot on the follower
# a = rear tube pivot -> yaw pin, b = yaw pin -> front tube pivot (each includes bars and the part's 158 mm span, whichever
# side the yaw pin is on). Steady state: the follower axle's instantaneous centre is the same O as the lead's, which
# fixes the follower pose from (psi_r, theta); psi_f is then whatever closes the chain. Positive angles = chain bends
# towards the inside of a left turn (same magnitude for a right turn).

def _wrap(a):
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def three_part_closure(R, a, b, p, psi_r, theta):
    """Steady-state closure on a circle of lead-axle radius R. psi_r, theta may be arrays.
    Returns (psi_f, delta, Of_x, Of_y, phi_f, valid): delta = follower heading relative to the lead (inside-positive),
    Of/phi_f = follower centre and heading in the lead frame, valid = False where the follower axle cannot be tangent."""
    psi_r = np.asarray(psi_r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    a2 = psi_r + theta
    fx = -p - a * np.cos(psi_r) - b * np.cos(a2)             # follower front pivot
    fy = a * np.sin(psi_r) + b * np.sin(a2)
    dx, dy = fx, fy - R
    d2 = dx * dx + dy * dy
    valid = d2 >= p * p
    rf = np.sqrt(np.maximum(d2 - p * p, 0.0))
    phi_f = np.arctan2(dy, dx) - np.arctan2(p, rf) + math.pi / 2.0
    delta = _wrap(-phi_f)
    psi_f = _wrap(delta - a2)
    return psi_f, delta, fx - p * np.cos(phi_f), fy - p * np.sin(phi_f), phi_f, valid


def chassis_overlap(of_x, of_y, phi_f, half_len, half_width, clearance=0.0):
    """True where the follower chassis rectangle overlaps the lead's (separating-axis test, lead at origin heading 0)."""
    hl, hw = half_len + clearance, half_width + clearance
    c, s = np.cos(phi_f), np.sin(phi_f)
    axes = ((np.ones_like(c), np.zeros_like(c)), (np.zeros_like(c), np.ones_like(c)), (c, s), (-s, c))
    separated = np.zeros(np.shape(c), dtype=bool)
    for ux, uy in axes:
        ra = hl * np.abs(ux) + hw * np.abs(uy)
        rb = hl * np.abs(ux * c + uy * s) + hw * np.abs(-ux * s + uy * c)
        separated |= np.abs(of_x * ux + of_y * uy) > ra + rb
    return ~separated


def _segment_hits_box(x0, y0, x1, y1, cx, cy, phi, half_len, half_width):
    """True where segment (x0,y0)-(x1,y1) touches the rectangle centred (cx,cy) with heading phi (slab test)."""
    c, s = np.cos(phi), np.sin(phi)
    ax, ay = c * (x0 - cx) + s * (y0 - cy), -s * (x0 - cx) + c * (y0 - cy)
    bx, by = c * (x1 - cx) + s * (y1 - cy), -s * (x1 - cx) + c * (y1 - cy)
    tmin, tmax = np.zeros(np.shape(ax)), np.ones(np.shape(ax))
    hit = np.ones(np.shape(ax), dtype=bool)
    for a0, d, h in ((ax, bx - ax, half_len), (ay, by - ay, half_width)):
        par = np.abs(d) < 1e-12
        hit &= ~par | (np.abs(a0) <= h)
        dd = np.where(par, 1.0, d)
        t1, t2 = (-h - a0) / dd, (h - a0) / dd
        tmin = np.where(par, tmin, np.maximum(tmin, np.minimum(t1, t2)))
        tmax = np.where(par, tmax, np.minimum(tmax, np.maximum(t1, t2)))
    return hit & (tmin <= tmax)


@dataclass(frozen=True)
class ThreePartHitch:
    pivot_m: float = 0.40              # tube pivot distance from the trailer centre (chassis half length - pivot inset)
    psi_max_deg: float = 85.0          # tube pivot limit: mechanical stop of the sector drive (Subdesigns/drawbar)
    theta_max_deg: float = 85.0        # vertical clevis pin limit (fork ears / pin geometry; confirm on the part)
    half_len_m: float = 0.45
    half_width_m: float = 0.25
    clearance_m: float = 0.02          # chassis-chassis keep-out
    bars_clear_other_chassis: bool = True   # bars may not sweep through the neighbouring trailer's chassis outline
    bars_in_chassis_plane: bool = False     # True: bars share the chassis height, so a bar may not cross its own chassis
                                            # (limits the tube pivot to atan(half_width / pivot inset))
    n_trailers: int = 5                     # chain length for the non-adjacent chassis check (1 or 2 = adjacent pair only)
    tracking_tol: float | None = 0.03       # follower axle radius must stay within this fraction of the lead's (path
                                            # tracking, as for the old equal-bar formula where R_f = R); None = ignore

    def feasible(self, R, a, b, margin=1.0, step_deg=1.0):
        """(ok, psi_r, theta, psi_f) -- best joint set at radius R (largest slack to the limits), or ok=False."""
        psi_max = self.psi_max_deg
        if self.bars_in_chassis_plane:
            psi_max = min(psi_max, math.degrees(math.atan2(self.half_width_m, self.half_len_m - self.pivot_m)))
        pm, tm = math.radians(psi_max) * margin, math.radians(self.theta_max_deg) * margin
        pr = np.radians(np.arange(-psi_max * margin, psi_max * margin + 1e-9, step_deg))
        th = np.radians(np.arange(-self.theta_max_deg * margin, self.theta_max_deg * margin + 1e-9, step_deg))
        PR, TH = np.meshgrid(pr, th, indexing="ij")
        pf, _, ox, oy, phi, valid = three_part_closure(R, a, b, self.pivot_m, PR, TH)
        slack = np.minimum(np.minimum(pm - np.abs(PR), tm - np.abs(TH)), pm - np.abs(pf))
        ok = valid & (slack >= 0.0) & ~chassis_overlap(ox, oy, phi, self.half_len_m, self.half_width_m, self.clearance_m)
        if self.n_trailers > 2 and ok.any():
            # Every hitch repeats the same joint set (tracking => same radius), so trailer k+1's pose follows from trailer k's.
            hl, hw = self.half_len_m + self.clearance_m, self.half_width_m + self.clearance_m
            poses = [(np.zeros_like(ox), np.zeros_like(ox), np.zeros_like(ox)), (ox, oy, phi)]
            for _ in range(self.n_trailers - 2):
                x, y, h = poses[-1]
                c, s_ = np.cos(h), np.sin(h)
                poses.append((x + c * ox - s_ * oy, y + s_ * ox + c * oy, h + phi))
            for i in range(len(poses)):
                for j in range(i + 2, len(poses)):
                    xi, yi, hi = poses[i]
                    xj, yj, hj = poses[j]
                    c, s_ = np.cos(hi), np.sin(hi)                     # trailer j in trailer i's frame
                    dx, dy = xj - xi, yj - yi
                    ok &= ~chassis_overlap(c * dx + s_ * dy, -s_ * dx + c * dy, hj - hi, self.half_len_m, self.half_width_m, self.clearance_m)
        if self.tracking_tol is not None:
            ok &= np.abs(np.hypot(ox, oy - R) - R) <= max(self.tracking_tol * R, 0.01)
        if self.bars_clear_other_chassis and ok.any():
            hl, hw = self.half_len_m + self.clearance_m, self.half_width_m + self.clearance_m
            px, yx, yy = -self.pivot_m, -self.pivot_m - a * np.cos(PR), a * np.sin(PR)
            fx, fy = yx - b * np.cos(PR + TH), yy + b * np.sin(PR + TH)
            ok &= ~_segment_hits_box(px, 0.0, yx, yy, ox, oy, phi, hl, hw)              # rear bar vs follower chassis
            ok &= ~_segment_hits_box(yx, yy, fx, fy, 0.0, 0.0, 0.0, hl, hw)             # front bar vs lead chassis
        if not ok.any():
            return False, 0.0, 0.0, 0.0
        i = np.unravel_index(np.argmax(np.where(ok, slack, -1.0)), ok.shape)
        return True, float(PR[i]), float(TH[i]), float(pf[i])

    def min_radius(self, a, b, margin=1.0, r_hi=10.0, r_lo=0.2, scan_m=0.05):
        """Tightest lead-axle radius R such that every radius from R up to r_hi is feasible, for fixed arm lengths.
        Scans downward from r_hi (feasibility is not monotonic once the whole chain's chassis outlines are checked, so
        bisection alone could return a radius with an infeasible gap above it), then bisects the first failing step.
        Returns inf if r_hi itself is infeasible."""
        if not self.feasible(r_hi, a, b, margin)[0]:
            return math.inf
        good, r = r_hi, r_hi - scan_m
        while r >= r_lo:
            if not self.feasible(r, a, b, margin)[0]:
                lo, hi = r, good
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (lo, mid) if self.feasible(mid, a, b, margin)[0] else (mid, hi)
                return hi
            good, r = r, r - scan_m
        return good

    def best_min_radius(self, a_range, b_range, margin=1.0, n=5):
        """Tightest radius over the adjustable arm lengths (variable bars shorten/lengthen as needed). (R, a, b)."""
        best = (math.inf, a_range[0], b_range[0])
        for a in np.linspace(a_range[0], a_range[1], n):
            for b in np.linspace(b_range[0], b_range[1], n):
                r = self.min_radius(float(a), float(b), margin)
                if r < best[0]:
                    best = (r, float(a), float(b))
        return best


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
    if getattr(cfg, "planner_hitch_model", "legacy") == "three_part" and getattr(cfg, "bar_yaw_control", False):
        a_rng = tuple(cfg.rear_bar_range_m) if var else (cfg.nominal_rear_bar_m,) * 2
        b_rng = tuple(cfg.front_bar_range_m) if var else (cfg.nominal_front_bar_m,) * 2
        p = cfg.chassis_half_len_m - cfg.bar_pivot_inset_m
        psi_deg = min(abs(cfg.rear_bar_yaw_range_deg[0]), abs(cfg.rear_bar_yaw_range_deg[1]),
                      abs(cfg.front_bar_yaw_range_deg[0]), abs(cfg.front_bar_yaw_range_deg[1]))
        hitch = ThreePartHitch(pivot_m=p, psi_max_deg=psi_deg, half_len_m=cfg.chassis_half_len_m,
                               half_width_m=cfg.chassis_half_width_m, n_trailers=cfg.num_trailers)
        R = hitch.best_min_radius(a_rng, b_rng)[0]
        return margin / R if math.isfinite(R) and R > 0 else 0.0
    if var:
        L_r, L_f = cfg.rear_bar_range_m[0], cfg.front_bar_range_m[0]
    else:
        L_r, L_f = cfg.nominal_rear_bar_m, cfg.nominal_front_bar_m
    psi = math.radians(min(abs(cfg.yaw_angle_range_deg[0]), abs(cfg.yaw_angle_range_deg[1]))) * margin
    return hitch_limited_curvature(L_r, L_f, psi)
