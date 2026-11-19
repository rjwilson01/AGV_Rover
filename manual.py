"""Manual set-points for --view: every actuator the controller normally drives can be set by hand from the keyboard.

Kept free of viewer code so it can be tested without a window: ManualControls holds the set-points, handles key codes
(GLFW codes, as passed to mujoco.viewer's key_callback), writes the actuator controls, and renders the help panel text.

Hot keys (also listed on screen in the viewer):
    Left / Right   previous / next set-point
    Up / Down      increase / decrease the selected set-point by its step
    Page Up / Down bigger / smaller step (x2 / x0.5)
    Backspace      zero the selected set-point (bars back to their longest)
    R              reset ALL set-points
    Space          pause / resume physics (paused: the model is only re-posed, actuators do nothing)
The viewer has its own hot keys too (F1 shows them); a few letters toggle viewer display flags, which is harmless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bar_geometry import trailer_cable_len
from BaseSnakeController import calc_cable_len

# GLFW key codes
KEY_SPACE, KEY_R, KEY_BACKSPACE = 32, 82, 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267

# Same corner order the controller uses for the four cables of a hitch: (name, y sign, z sign)
CORNER_SIGNS = (("tl", 1.0, 1.0), ("tr", -1.0, 1.0), ("bl", 1.0, -1.0), ("br", -1.0, -1.0))   # same as main.py


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
        self.params: list[Param] = []
        self.sel = 0
        yl, yh = cfg.yaw_angle_range_deg
        r_min, r_max = cfg.rear_bar_range_m
        f_min, f_max = cfg.front_bar_range_m
        self.params.append(Param("wheel_speed", "all wheels speed", "m/s", 0.0, 0.0, -3.0, 3.0, 0.05))
        for j in range(1, cfg.num_trailers):
            self.params += [
                Param(f"h{j}_yaw", f"hitch {j} yaw", "deg", 0.0, 0.0, float(yl), float(yh), 1.0),
                Param(f"h{j}_pitch", f"hitch {j} ball pitch", "deg", 0.0, 0.0, -30.0, 30.0, 1.0),
                Param(f"h{j}_roll", f"hitch {j} ball roll", "deg", 0.0, 0.0, -30.0, 30.0, 1.0),
                Param(f"h{j}_rear", f"hitch {j} rear bar length", "m", r_max, r_max, r_min, r_max, 0.02),
                Param(f"h{j}_front", f"hitch {j} front bar length", "m", f_max, f_max, f_min, f_max, 0.02),
            ]

    # -- values ---------------------------------------------------------------------------------------------------
    def get(self, key: str) -> float:
        return next(p.value for p in self.params if p.key == key)

    def reset_all(self) -> None:
        for p in self.params:
            p.value = p.default

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
        else:
            return False
        return True

    # -- actuators ------------------------------------------------------------------------------------------------
    def apply(self, data, model_actuators: set[str]) -> None:
        """Write the set-points to the actuators, mirroring what main.py does from the controller's outputs."""
        cfg = self.cfg
        _, r_max = cfg.rear_bar_range_m
        _, f_max = cfg.front_bar_range_m
        v = self.get("wheel_speed") / cfg.wheel_radius_m
        for i in range(1, cfg.num_trailers + 1):
            for side in ("l", "r"):
                data.actuator(f"m_t{i}_{side}").ctrl = v
        rad = np.deg2rad
        for j in range(1, cfg.num_trailers):
            y = rad(self.get(f"h{j}_yaw"))
            p = rad(self.get(f"h{j}_pitch"))
            r = rad(self.get(f"h{j}_roll"))
            r_len, f_len = self.get(f"h{j}_rear"), self.get(f"h{j}_front")
            rear_act, front_act = f"act_slide_rear_{j}", f"act_slide_front_{j + 1}"
            if rear_act in model_actuators:
                data.actuator(rear_act).ctrl = r_max - r_len
            if front_act in model_actuators:
                data.actuator(front_act).ctrl = f_max - f_len
            if self.bars:
                # tubes take half the yaw each (inside their own ranges); the ball takes the rest
                psi_r = float(np.clip(y / 2.0, rad(cfg.rear_bar_yaw_range_deg[0]), rad(cfg.rear_bar_yaw_range_deg[1])))
                psi_f = float(np.clip(y / 2.0, rad(cfg.front_bar_yaw_range_deg[0]), rad(cfg.front_bar_yaw_range_deg[1])))
                y_ball = y - psi_r - psi_f
                data.actuator(f"act_yaw_rear_{j}").ctrl = psi_r
                data.actuator(f"act_yaw_front_{j + 1}").ctrl = psi_f
                for k, (code, ys, zs) in enumerate(CORNER_SIGNS):
                    data.actuator(f"cap_t{j}_t{j + 1}_{code}").ctrl = (
                        trailer_cable_len(psi_r, psi_f, (y_ball, p, r), ys, zs, cfg, r_len, f_len)
                        - cfg.cable_preload_offset_m
                    )
            else:
                for k, (code, ys, zs) in enumerate(CORNER_SIGNS):
                    data.actuator(f"cap_t{j}_t{j + 1}_{code}").ctrl = (
                        calc_cable_len(-y, -p, r, ys, zs, cfg, rear_bar_m=r_len, front_bar_m=f_len)
                        - cfg.cable_preload_offset_m
                    )

    # -- on-screen text -------------------------------------------------------------------------------------------
    def panel(self) -> tuple[str, str]:
        """(left column, right column) text for the viewer overlay."""
        head = [
            "MANUAL CONTROLS" + ("   [PHYSICS PAUSED]" if self.paused else ""),
            "Left/Right  select set-point",
            "Up/Down     increase / decrease",
            "PgUp/PgDn   bigger / smaller step",
            "Backspace   zero selected    R  reset all",
            "Space       pause / resume physics",
            "",
        ]
        left = list(head)
        right = [""] * len(head)
        for i, p in enumerate(self.params):
            left.append(("> " if i == self.sel else "  ") + p.label)
            right.append(f"{p.value:+8.2f} {p.unit}  (step {p.step * p.scale:g}, {p.lo:g}..{p.hi:g})")
        return "\n".join(left), "\n".join(right)
