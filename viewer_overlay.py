import mujoco
import numpy as np

# Overlay visual constants
OVERLAY_Z_HEIGHT_M = 0.02
WAYPOINT_RADIUS_M = 0.08
LOOKAHEAD_RADIUS_M = 0.10
LINE_WIDTH_PX = 4.0

WAYPOINT_RGBA = np.array([1.0, 0.85, 0.10, 0.85], dtype=np.float32)   # Yellow
PATH_LINE_RGBA = np.array([0.20, 0.90, 0.40, 0.70], dtype=np.float32)  # Green
LOOKAHEAD_RGBA = np.array([1.0, 0.25, 0.25, 0.95], dtype=np.float32)  # Red


def render_path_overlay(viewer, waypoints, lookahead_pt=None) -> None:
    """Draw waypoint spheres, connecting path segments, and the active lookahead target."""
    scn = viewer.user_scn
    scn.ngeom = 0
    pts = np.asarray(waypoints, dtype=float)
    identity_mat = np.eye(3, dtype=np.float64).flatten()

    # 1. Waypoint markers & path segments
    for i, pt in enumerate(pts):
        if scn.ngeom >= scn.maxgeom:
            break
        pos = np.array([pt, pt, OVERLAY_Z_HEIGHT_M], dtype=np.float64)
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([WAYPOINT_RADIUS_M, 0.0, 0.0], dtype=np.float64),
            pos,
            identity_mat,
            WAYPOINT_RGBA,
        )
        scn.ngeom += 1

        if i < len(pts) - 1 and scn.ngeom < scn.maxgeom:
            nxt = pts[i + 1]
            p1 = np.array([pt, pt, OVERLAY_Z_HEIGHT_M], dtype=np.float64)
            p2 = np.array([nxt, nxt, OVERLAY_Z_HEIGHT_M], dtype=np.float64)
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                LINE_WIDTH_PX * 0.004,
                p1,
                p2,
            )
            scn.geoms[scn.ngeom].rgba[:] = PATH_LINE_RGBA
            scn.ngeom += 1

    # 2. Active Pure Pursuit lookahead target
    if lookahead_pt is not None and scn.ngeom < scn.maxgeom:
        la_pos = np.array([lookahead_pt, lookahead_pt, OVERLAY_Z_HEIGHT_M + 0.02], dtype=np.float64)
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([LOOKAHEAD_RADIUS_M, 0.0, 0.0], dtype=np.float64),
            la_pos,
            identity_mat,
            LOOKAHEAD_RGBA,
        )
        scn.ngeom += 1
