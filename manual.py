"""State for the --view sandbox: set-points, algorithm switches, key handling and the text panel.

No viewer / physics / Tk code here, so it can be tested without a window. The physics side is in sandbox.py and the
pop-out window in tkpanel.py.

SET-POINTS are numbers you control. ALGORITHM SWITCHES decide, for each stage of the real controller, whether the
controller's own output is used (ON) or your manual set-point (OFF):

    wheel_law      ON: lead wheels from lead speed + curvature, follower wheels from the hitch kinematics + measured
                       hitch yaw (the real follower law).   OFF: every wheel at the manual 'all wheels speed'.
    bar_len_law    ON: bar lengths from the Variable controller (tightness law, driven by the curvature the followers see).
                   OFF: manual rear/front bar length set-points.
    bar_yaw_law    ON: tube yaw from the controller's feed-forward (arctan(length * curvature)).
                   OFF: the manual hitch-yaw set-point (split between the two tubes, ball takes the rest).
    pitch_limit    ON: bars shorten when the hitch pitch would put a bar end on the ground.
    tube_geometry  ON: follower wheel law uses the ball's true off-axis position from the tube yaws.
    cable_measured ON: cables sized for the MEASURED tube yaws and bar lengths.  OFF: for the commanded ones.
    ball_hold_ir   ON: ball pitch/roll is held at the value measured by the IR camera (ignores manual pitch/roll).
                   OFF: the manual ball pitch/roll set-points are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# GLFW key codes (MuJoCo viewer) -- the Tk window maps its own keysyms onto these
KEY_SPACE, KEY_R, KEY_BACKSPACE = 32, 82, 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267

# (name, default, label, cfg field it toggles or None)
FLAGS = (
    ("wheel_law", True, "Wheel-speed law (follower speeds from hitch kinematics)", None),
    ("bar_len_law", True, "Bar-length law (tightness from curvature)", None),
    ("bar_yaw_law", False, "Tube-yaw law (feed-forward from curvature)", None),
    ("pitch_limit", True, "Pitch limit (shorten bars near the ground)", "bar_pitch_limit"),
    ("tube_geometry", True, "Wheel law uses tube/ball geometry", "wheel_use_tube_geometry"),
    ("cable_measured", True, "Cables sized for MEASURED tubes/bars", "cable_use_measured_tubes"),
    ("ball_hold_ir", False, "Ball pitch/roll held by IR (off: manual)", "use_ir_camera_feedback"),
)


@dataclass
class Param:
    key: str            # short id, e.g. "h1_yaw"
    label: str          # shown on screen
    unit: str
    value: float
    default: float
    lo: float
    hi: float
    step: float         # base step per key press
    scale: float = 1.0  # multiplier applied to the step (Page Up/Down)

    def nudge(self, direction: int) -> None:
        self.value = float(np.clip(self.value + direction * self.step * self.scale, self.lo, self.hi))


class ManualControls:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bars = bool(cfg.bar_yaw_control)
        self.paused = False
        self.pause_on_compression = bool(cfg.view_pause_on_compression)
        self.alert = ""                                  # why the physics was paused automatically
        self.params: list[Param] = []
        self.sel = 0
        self.flags = {name: default for name, default, _, _ in FLAGS}
        for name, _, _, field in FLAGS:                      # a flag that mirrors a config field starts from the config
            if field is not None and hasattr(cfg, field):
                self.flags[name] = bool(getattr(cfg, field)) if name != "ball_hold_ir" else False
        yl, yh = cfg.yaw_angle_range_deg
        r_min, r_max = cfg.rear_bar_range_m
        f_min, f_max = cfg.front_bar_range_m
        kmax = float(cfg.max_curvature)
        self.params += [
            Param("lead_speed", "lead speed", "m/s", float(cfg.view_lead_speed_mps), float(cfg.view_lead_speed_mps), -2.0, 2.0, 0.05),
            Param("lead_kappa", "lead curvature", "1/m", 0.0, 0.0, -kmax, kmax, 0.02),
            Param("follow_kappa", "curvature the followers see", "1/m", 0.0, 0.0, -kmax, kmax, 0.02),
            Param("wheel_speed", "all wheels speed (wheel law OFF)", "m/s", 0.0, 0.0, -3.0, 3.0, 0.05),
        ]
        for j in range(1, cfg.num_trailers):
            self.params += [
                Param(f"h{j}_yaw", f"hitch {j} yaw (yaw law OFF)", "deg", 0.0, 0.0, float(yl), float(yh), 1.0),
                Param(f"h{j}_pitch", f"hitch {j} ball pitch", "deg", 0.0, 0.0, -30.0, 30.0, 1.0),
                Param(f"h{j}_roll", f"hitch {j} ball roll", "deg", 0.0, 0.0, -30.0, 30.0, 1.0),
                Param(f"h{j}_rear", f"hitch {j} rear bar (len law OFF)", "m", r_max, r_max, r_min, r_max, 0.02),
                Param(f"h{j}_front", f"hitch {j} front bar (len law OFF)", "m", f_max, f_max, f_min, f_max, 0.02),
            ]
        self.readout = ""        # filled by the sandbox each frame: commanded vs measured values

    def copy_state_from(self, old: "ManualControls") -> None:
        """Carry the set-point values, switches and selection over after an XML reload (matched by key)."""
        vals = {p.key: p for p in old.params}
        for p in self.params:
            if p.key in vals:
                p.value = float(np.clip(vals[p.key].value, p.lo, p.hi))
                p.scale = vals[p.key].scale
        self.flags.update({k: v for k, v in old.flags.items() if k in self.flags})
        self.paused = old.paused
        self.pause_on_compression = old.pause_on_compression
        self.sel = min(old.sel, len(self.params) - 1)

    # -- values ---------------------------------------------------------------------------------------------------
    def get(self, key: str) -> float:
        return next(p.value for p in self.params if p.key == key)

    def reset_all(self) -> None:
        for p in self.params:
            p.value = p.default

    def toggle(self, name: str) -> bool:
        self.flags[name] = not self.flags[name]
        return self.flags[name]

    def cfg_overrides(self) -> dict:
        """Config fields implied by the switches (applied by the sandbox with dataclasses.replace)."""
        return {field: self.flags[name] for name, _, _, field in FLAGS if field is not None}

    # -- keys -----------------------------------------------------------------------------------------------------
    def key(self, code: int) -> bool:
        """Handle a key press; True if it was one of ours."""
        n = len(self.params)
        if code == KEY_RIGHT:
            self.sel = (self.sel + 1) % n
        elif code == KEY_LEFT:
            self.sel = (self.sel - 1) % n
        elif code == KEY_UP:
            self.params[self.sel].nudge(+1)
        elif code == KEY_DOWN:
            self.params[self.sel].nudge(-1)
        elif code == KEY_PAGE_UP:
            self.params[self.sel].scale = min(self.params[self.sel].scale * 2.0, 32.0)
        elif code == KEY_PAGE_DOWN:
            self.params[self.sel].scale = max(self.params[self.sel].scale * 0.5, 1.0 / 16.0)
        elif code == KEY_BACKSPACE:
            self.params[self.sel].value = self.params[self.sel].default
        elif code == KEY_R:
            self.reset_all()
        elif code == KEY_SPACE:
            self.paused = not self.paused
            if not self.paused:
                self.alert = ""
        else:
            return False
        return True

    # -- on-screen text (viewer overlay; the Tk window shows the same information as widgets) ----------------------
    def panel(self) -> tuple[str, str]:
        head = [
            "MANUAL SANDBOX" + ("   [PHYSICS PAUSED]" if self.paused else ""),
            *([self.alert] if self.alert else []),
            "Left/Right select | Up/Down +/- | PgUp/PgDn step | Backspace zero | R reset | Space pause",
            "(the pop-out window has the same controls plus switches for each algorithm)",
            "",
        ]
        left = list(head)
        right = [""] * len(head)
        for name, _, label, _ in FLAGS:
            left.append(f"[{'x' if self.flags[name] else ' '}] {label}")
            right.append("")
        left.append("")
        right.append("")
        for i, p in enumerate(self.params):
            left.append(("> " if i == self.sel else "  ") + p.label)
            right.append(f"{p.value:+8.2f} {p.unit}  (step {p.step * p.scale:g}, {p.lo:g}..{p.hi:g})")
        return "\n".join(left), "\n".join(right)
