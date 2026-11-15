import mujoco
import numpy as np

# Overlay visual constants
OVERLAY_Z_HEIGHT_M = 0.04
WAYPOINT_RADIUS_M = 0.08
LOOKAHEAD_RADIUS_M = 0.11
PATH_CAPSULE_RADIUS_M = 0.015

WAYPOINT_RGBA = np.array([1.0, 0.85, 0.10, 0.90], dtype=np.float32)   # Yellow
PATH_LINE_RGBA = np.array([0.10, 0.80, 1.00, 0.70], dtype=np.float32)  # Cyan
LOOKAHEAD_RGBA = np.array([1.0, 0.20, 0.10, 0.95], dtype=np.float32)  # Red-Orange


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
