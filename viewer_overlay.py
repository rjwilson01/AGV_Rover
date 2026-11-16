import mujoco
import numpy as np

from bar_geometry import pivot_x_m

# Overlay visual constants
OVERLAY_Z_HEIGHT_M = 0.04
WAYPOINT_RADIUS_M = 0.08
LOOKAHEAD_RADIUS_M = 0.11
PATH_CAPSULE_RADIUS_M = 0.015

WAYPOINT_RGBA = np.array([1.0, 0.85, 0.10, 0.90], dtype=np.float32)   # Yellow
PATH_LINE_RGBA = np.array([0.10, 0.80, 1.00, 0.70], dtype=np.float32)  # Cyan
LOOKAHEAD_RGBA = np.array([1.0, 0.20, 0.10, 0.95], dtype=np.float32)  # Red-Orange

BAR_CMD_RGBA = np.array([1.0, 0.55, 0.0, 0.95], dtype=np.float32)      # Orange: commanded bar direction
BAR_MEAS_RGBA = np.array([0.2, 1.0, 0.3, 0.95], dtype=np.float32)      # Green: actual pivot -> ball line
BAR_CAPSULE_RADIUS_M = 0.012
BAR_OVERLAY_LIFT_M = 0.07


def render_path_overlay(viewer, waypoints, lookahead_pt=None) -> None:
    """Draw waypoint spheres, connecting path capsules, and the active lookahead target."""
    scn = viewer.user_scn
    scn.ngeom = 0
    pts = np.asarray(waypoints, dtype=float)
    identity_mat = np.eye(3, dtype=np.float64).flatten()
    zero_size = np.zeros(3, dtype=np.float64)
    zero_pos = np.zeros(3, dtype=np.float64)

    # 1. Waypoint markers & path segments
    for i, pt in enumerate(pts):
        if scn.ngeom >= scn.maxgeom - 2:
            break
        x, y = float(pt[0]), float(pt[1])
        pos = np.array([x, y, OVERLAY_Z_HEIGHT_M + 0.01], dtype=np.float64)
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([WAYPOINT_RADIUS_M, 0.0, 0.0], dtype=np.float64),
            pos,
            identity_mat,
            WAYPOINT_RGBA,
        )
        scn.ngeom += 1

        if i < len(pts) - 1:
            nx, ny = float(pts[i + 1][0]), float(pts[i + 1][1])
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                zero_size,
                zero_pos,
                identity_mat,
                PATH_LINE_RGBA,
            )
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                PATH_CAPSULE_RADIUS_M,
                np.array([x, y, OVERLAY_Z_HEIGHT_M], dtype=np.float64),
                np.array([nx, ny, OVERLAY_Z_HEIGHT_M], dtype=np.float64),
            )
            scn.ngeom += 1

    # 2. Active Pure Pursuit lookahead target
    if lookahead_pt is not None and scn.ngeom < scn.maxgeom:
        lx, ly = float(lookahead_pt[0]), float(lookahead_pt[1])
        la_pos = np.array([lx, ly, OVERLAY_Z_HEIGHT_M + 0.11], dtype=np.float64)
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([LOOKAHEAD_RADIUS_M, 0.0, 0.0], dtype=np.float64),
            la_pos,
            identity_mat,
            LOOKAHEAD_RGBA,
        )
        scn.ngeom += 1


def _add_line(scn, p0, p1, rgba, radius: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64).flatten(), rgba,
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
    scn.ngeom += 1


def render_bar_yaw_overlay(viewer, data, cfg, bar_yaws, bar_lengths) -> None:
    """Yaw-tube builds: draw each drawbar's commanded direction (orange) and the actual pivot-to-ball
    line (green) at every hitch. Call after render_path_overlay(), which resets the user scene.

    Commanded directions follow the tube yaw convention in bar_geometry.py: the rear bar points back from
    trailer j rotated by psi_r; the front bar points forward from trailer j+1 rotated by -psi_f.
    """
    scn = viewer.user_scn
    lift = np.array([0.0, 0.0, BAR_OVERLAY_LIFT_M])
    p = pivot_x_m(cfg)
    for j in range(1, cfg.num_trailers):
        psi_r, psi_f = bar_yaws[j - 1]
        r_cmd, f_cmd = bar_lengths[j - 1]
        rot_j = data.body(f"trailer_{j}").xmat.reshape(3, 3)
        rot_n = data.body(f"trailer_{j + 1}").xmat.reshape(3, 3)
        rear_pivot = np.array(data.body(f"t{j}_rear_tube").xpos) + lift
        front_pivot = np.array(data.body(f"t{j + 1}_front_tube").xpos) + lift
        ball = np.array(data.body(f"t{j + 1}_front_socket").xpos) + lift

        rear_dir = rot_j @ np.array([-np.cos(psi_r), -np.sin(psi_r), 0.0])
        front_dir = rot_n @ np.array([np.cos(psi_f), -np.sin(psi_f), 0.0])
        _add_line(scn, rear_pivot, rear_pivot + rear_dir * (r_cmd - p), BAR_CMD_RGBA, BAR_CAPSULE_RADIUS_M * 1.6)
        _add_line(scn, front_pivot, front_pivot + front_dir * (f_cmd - p), BAR_CMD_RGBA, BAR_CAPSULE_RADIUS_M * 1.6)
        _add_line(scn, rear_pivot, ball, BAR_MEAS_RGBA, BAR_CAPSULE_RADIUS_M)
        _add_line(scn, front_pivot, ball, BAR_MEAS_RGBA, BAR_CAPSULE_RADIUS_M)
