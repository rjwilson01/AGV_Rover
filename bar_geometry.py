"""Geometry of the yaw-controlled drawbars (Variable build with cfg.bar_yaw_control = true).

Each trailer end has a tube that pivots (yaw hinge, axis +Z) near the chassis edge. The drawbar
slides inside the tube and ends in the passive ball joint. In the real build two horizontal cables
(pull/pull) set the tube's yaw; the simulation commands the hinge angle directly. Per hitch, between trailer j (ahead) and
trailer j+1 (behind), the kinematic chain is

    X = Tr(-p) Rz(psi_r) Tr(-s_r) Rb Tr(-s_f) Rz(psi_f) Tr(-p)

(trailer j+1 frame expressed in trailer j frame) where
    p       pivot distance from the trailer centre (chassis half length minus bar_pivot_inset_m)
    s_r/s_f pivot-to-ball reach of the rear / front bar  (bar length from trailer centre minus p)
    psi_r   rear tube hinge angle (rotates the rear bar about trailer j's rear pivot)
    psi_f   front tube hinge angle (rotates trailer j+1 about its front pivot, relative to the tube)
    Rb      passive ball joint rotation Rz(yaw) Ry(pitch) Rx(roll)

With Rb = I the two bars form one straight line and the trailer-to-trailer yaw is psi_r + psi_f.
Bar lengths r, f follow the existing meaning: distance from the trailer centre to the ball when
the bar is straight (psi = 0).

All angles in radians.
"""

import numpy as np

from config import AGVConfig

def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ball_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation matrix for the ball joint, same ZYX convention as quat_to_euler()."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def euler_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) of a rotation matrix, inverse of ball_rotation()."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


def pivot_x_m(cfg: AGVConfig) -> float:
    """Distance of the tube pivot from the trailer centre (along the trailer axis)."""
    return cfg.chassis_half_len_m - cfg.bar_pivot_inset_m


def validate_bar_config(cfg: AGVConfig) -> None:
    """Raise ValueError if the bar/tube dimensions cannot be built."""
    p = pivot_x_m(cfg)
    if p <= 0.0:
        raise ValueError(f"bar_pivot_inset_m={cfg.bar_pivot_inset_m} puts the pivot at or past the trailer centre")
    for name, (b_min, _) in (("rear", cfg.rear_bar_range_m), ("front", cfg.front_bar_range_m)):
        if b_min - p <= 0.05:
            raise ValueError(
                f"{name}_bar_range_m minimum {b_min} leaves only {b_min - p:.3f} m of bar beyond the pivot at {p:.3f} m"
            )


_PRINTED_WARNINGS: set[str] = set()


def warn_once(tag: str, msg: str) -> None:
    """Print "[tag] WARNING: msg" the first time only. The viewer and config session rebuild the model on every change, so
    unguarded prints repeat dozens of times; the viewer's model panel (viewmodel.py) still lists the current warnings."""
    key = f"{tag}|{msg}"
    if key not in _PRINTED_WARNINGS:
        _PRINTED_WARNINGS.add(key)
        print(f"[{tag}] WARNING: {msg}")


def bar_packaging_warnings(cfg: AGVConfig) -> list[str]:
    """Mechanical sanity checks on the telescoping bars (tube outside the chassis, rod slides in from the ball end)."""
    p = pivot_x_m(cfg)
    out = []
    for name, (b_min, b_max) in (("rear", cfg.rear_bar_range_m), ("front", cfg.front_bar_range_m)):
        tube = max(0.05, b_min - p - 0.02)
        stroke = b_max - b_min
        engage = tube - stroke
        if stroke > 0 and engage < cfg.bar_min_engagement_m:
            out.append(
                f"{name} bar: only {engage * 100:.0f} cm of rod stays inside the tube at full extension "
                f"(tube {tube:.2f} m, stroke {stroke:.2f} m, want >= {cfg.bar_min_engagement_m * 100:.0f} cm)"
            )
        housed = 2.0 * (stroke + cfg.bar_min_engagement_m)
        if stroke > 0 and housed > 2.0 * cfg.chassis_half_len_m:
            out.append(
                f"{name} bar: housing the tubes inside the chassis would need {housed:.2f} m for both ends but the "
                f"chassis is {2 * cfg.chassis_half_len_m:.2f} m long: the rods would meet. Keep the tubes outside."
            )
    return out


def hitch_transform(
    psi_r: float,
    psi_f: float,
    ball_ypr: tuple[float, float, float],
    r: float,
    f: float,
    cfg: AGVConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose (R, t) of trailer j+1's frame in trailer j's frame."""
    p = pivot_x_m(cfg)
    s_r, s_f = r - p, f - p
    Rb = ball_rotation(*ball_ypr)
    Rz_r, Rz_f = rot_z(psi_r), rot_z(psi_f)

    t = np.array([-p, 0.0, 0.0]) + Rz_r @ np.array([-s_r, 0.0, 0.0])      # ball position in trailer j frame
    R = Rz_r @ Rb                                                          # front bar frame orientation
    t = t + R @ np.array([-s_f, 0.0, 0.0])                                 # front pivot
    R = R @ Rz_f                                                           # trailer j+1 orientation
    t = t + R @ np.array([-p, 0.0, 0.0])                                   # trailer j+1 centre
    return R, t


def trailer_cable_len(
    psi_r: float,
    psi_f: float,
    ball_ypr: tuple[float, float, float],
    y_sign: float,
    z_sign: float,
    cfg: AGVConfig,
    r: float,
    f: float,
) -> float:
    """Length of the trailer-to-trailer cable (corner y_sign, z_sign) for the full bar geometry.

    Generalises calc_cable_len(): with psi_r = psi_f = 0 and ball_ypr = (y, p, r) it equals
    calc_cable_len(-y, -p, r, ...).
    """
    R, t = hitch_transform(psi_r, psi_f, ball_ypr, r, f, cfg)
    site_front = np.array([cfg.chassis_half_len_m, y_sign * cfg.cable_anchor_y_m, z_sign * cfg.cable_anchor_z_m])
    site_rear = np.array([-cfg.chassis_half_len_m, y_sign * cfg.cable_anchor_y_m, z_sign * cfg.cable_anchor_z_m])
    return float(np.linalg.norm(R @ site_front + t - site_rear))


def axle_distance(
    psi_r: float,
    psi_f: float,
    ball_ypr: tuple[float, float, float],
    r: float,
    f: float,
    cfg: AGVConfig,
) -> float:
    """Distance between the centres of trailer j and j+1 implied by the bar geometry."""
    _, t = hitch_transform(psi_r, psi_f, ball_ypr, r, f, cfg)
    return float(np.linalg.norm(t))


def ball_ypr_from_trailer_relative(
    trailer_ypr: tuple[float, float, float],
    psi_r: float,
    psi_f: float,
) -> tuple[float, float, float]:
    """Recover the ball joint rotation from the trailer-to-trailer R/P/Y (what the IR camera sees)
    and the two measured tube angles:  R_trailer = Rz(psi_r) Rb Rz(psi_f)."""
    R_t = ball_rotation(*trailer_ypr)
    Rb = rot_z(-psi_r) @ R_t @ rot_z(-psi_f)
    return euler_from_matrix(Rb)
