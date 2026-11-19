"""Pop-out Tk control window for the --view sandbox (algorithm switches, set-points, live readout).

Keys work while this window has focus and auto-repeat when held (Windows/macOS/Linux Tk all deliver repeated key presses);
the +/- buttons also repeat while the mouse button is held. The simulation loop calls update() every frame (no mainloop,
so there are no threads). Closing the window just hides it; the viewer keeps running.
"""
from __future__ import annotations

import manual as M

KEYSYMS = {
    "Up": M.KEY_UP, "Down": M.KEY_DOWN, "Left": M.KEY_LEFT, "Right": M.KEY_RIGHT,
    "Prior": M.KEY_PAGE_UP, "Next": M.KEY_PAGE_DOWN, "BackSpace": M.KEY_BACKSPACE,
    "r": M.KEY_R, "R": M.KEY_R, "space": M.KEY_SPACE,
    "plus": M.KEY_UP, "equal": M.KEY_UP, "KP_Add": M.KEY_UP,
    "minus": M.KEY_DOWN, "KP_Subtract": M.KEY_DOWN,
}
REPEAT_FIRST_MS, REPEAT_MS, REFRESH_S = 350, 60, 0.1


class ControlWindow:
    def __init__(self, ctl, tk_module=None, sandbox=None):
        if tk_module is None:
            import tkinter as tk_module            # may raise ImportError: the caller falls back to the viewer text panel
        self.tk = tk_module
        tk = self.tk
        self.ctl = ctl
        self.alive = True
        self._after_id = None
        self._last_refresh = 0.0
        self.root = tk.Tk()
        self.root.title("AGV manual sandbox")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("<KeyPress>", self._on_key)

        # --- algorithm switches
        sw = tk.LabelFrame(self.root, text="Algorithms (checked = the controller's output is used, unchecked = your set-point)")
        sw.pack(fill="x", padx=6, pady=4)
        self.flag_vars = {}
        for name, _, label, _ in M.FLAGS:
            var = tk.BooleanVar(value=ctl.flags[name])
            self.flag_vars[name] = var
            tk.Checkbutton(sw, text=label, variable=var, anchor="w",
                           command=lambda n=name: self._set_flag(n)).pack(fill="x")

        # --- set-points
        sp = tk.LabelFrame(self.root, text="Set-points (click a row or use Left/Right; Up/Down or +/- change it, hold to repeat)")
        sp.pack(fill="x", padx=6, pady=4)
        self.rows = []
        for i, p in enumerate(ctl.params):
            name = tk.Label(sp, text=p.label, anchor="w", width=38)
            name.grid(row=i, column=0, sticky="w")
            val = tk.Label(sp, text="", anchor="e", width=52, font=("Courier", 10))
            val.grid(row=i, column=1)
            btn_m = tk.Button(sp, text=" - ", width=3)
            btn_p = tk.Button(sp, text=" + ", width=3)
            btn_m.grid(row=i, column=2)
            btn_p.grid(row=i, column=3)
            for w in (name, val):
                w.bind("<Button-1>", lambda e, k=i: self._select(k))
            self._hold(btn_m, i, -1)
            self._hold(btn_p, i, +1)
            self.rows.append((name, val))

        # --- actions
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=6)
        tk.Button(bar, text="Zero selected (Backspace)", command=lambda: self._key(M.KEY_BACKSPACE)).pack(side="left")
        tk.Button(bar, text="Reset all (R)", command=lambda: self._key(M.KEY_R)).pack(side="left")
        self.pause_btn = tk.Button(bar, text="Pause physics (Space)", command=lambda: self._key(M.KEY_SPACE))
        self.pause_btn.pack(side="left")
        tk.Button(bar, text="Step bigger (PgUp)", command=lambda: self._key(M.KEY_PAGE_UP)).pack(side="left")
        tk.Button(bar, text="Step smaller (PgDn)", command=lambda: self._key(M.KEY_PAGE_DOWN)).pack(side="left")

        # --- live readout
        self.text = tk.Text(self.root, height=22, width=100, font=("Courier", 9))
        self.text.pack(fill="both", expand=True, padx=6, pady=4)
        self.forces = ForceWindow(self.root, sandbox, tk) if sandbox is not None else None
        self.refresh(force=True)

    # -- input ----------------------------------------------------------------------------------------------------
    def _on_key(self, event) -> None:
        code = KEYSYMS.get(getattr(event, "keysym", ""))
        if code is not None:
            self._key(code)

    def _key(self, code: int) -> None:
        self.ctl.key(code)
        self.refresh(force=True)

    def _select(self, i: int) -> None:
        self.ctl.sel = i
        self.refresh(force=True)

    def _set_flag(self, name: str) -> None:
        self.ctl.flags[name] = bool(self.flag_vars[name].get())

    def _hold(self, button, i: int, direction: int) -> None:
        """Press: act once, then repeat while the mouse button is held."""
        def tick():
            self.ctl.sel = i
            self.ctl.params[i].nudge(direction)
            self.refresh(force=True)
            self._after_id = self.root.after(REPEAT_MS, tick)

        def press(_e):
            self.ctl.sel = i
            self.ctl.params[i].nudge(direction)
            self.refresh(force=True)
            self._after_id = self.root.after(REPEAT_FIRST_MS, tick)

        def release(_e):
            if self._after_id is not None:
                self.root.after_cancel(self._after_id)
                self._after_id = None

        button.bind("<ButtonPress-1>", press)
        button.bind("<ButtonRelease-1>", release)

    def _close(self) -> None:
        self.alive = False
        try:
            self.root.destroy()
        except Exception:
            pass

    # -- display --------------------------------------------------------------------------------------------------
    def refresh(self, force: bool = False, now: float | None = None) -> None:
        if not self.alive:
            return
        import time
        now = time.time() if now is None else now
        if not force and now - self._last_refresh < REFRESH_S:
            return
        self._last_refresh = now
        for name, var in self.flag_vars.items():          # keep the boxes in step with any other change
            if var.get() != self.ctl.flags[name]:
                var.set(self.ctl.flags[name])
        for i, (p, (name, val)) in enumerate(zip(self.ctl.params, self.rows)):
            sel = i == self.ctl.sel
            name.config(bg="#ffe9a8" if sel else self.root.cget("bg"), text=("> " if sel else "  ") + p.label)
            val.config(text=f"{p.value:+8.3f} {p.unit}   step {p.step * p.scale:g}   [{p.lo:g}..{p.hi:g}]",
                       bg="#ffe9a8" if sel else self.root.cget("bg"))
        self.pause_btn.config(text="Resume physics (Space)" if self.ctl.paused else "Pause physics (Space)")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.ctl.readout or "(readout appears once the simulation is running)")
        if self.forces is not None:
            self.forces.refresh()

    def update(self) -> bool:
        """Call every frame. False once the window has been closed."""
        if not self.alive:
            return False
        try:
            self.refresh()
            self.root.update()
        except Exception:
            self.alive = False
        return self.alive


class ForceWindow:
    """Second window: the live force diagram of one hitch (see forceview.py). Buttons pick the hitch, tare the resting
    cable sum, and change the arrow scale."""

    def __init__(self, parent, sandbox, tk_module):
        tk = tk_module
        self.sb = sandbox
        self.hitch = 1
        self.scale = 0.25
        self.w, self.h = 760, 440
        self.win = tk.Toplevel(parent)
        self.win.title("AGV hitch forces (live)")
        bar = tk.Frame(self.win)
        bar.pack(fill="x")
        self.n_hitches = max(sandbox.cfg.num_trailers - 1, 1)
        for j in range(1, self.n_hitches + 1):
            tk.Button(bar, text=f"Hitch {j}", command=lambda jj=j: self.select(jj)).pack(side="left")
        tk.Button(bar, text="Tare resting tension", command=self.sb.tare).pack(side="left", padx=8)
        tk.Button(bar, text="Arrows x2", command=lambda: self.zoom(2.0)).pack(side="left")
        tk.Button(bar, text="Arrows /2", command=lambda: self.zoom(0.5)).pack(side="left")
        self.canvas = tk.Canvas(self.win, width=self.w, height=self.h, bg="white")
        self.canvas.pack(fill="both", expand=True)
        self._last = 0.0

    def select(self, j: int) -> None:
        self.hitch = j
        self.refresh(force=True)

    def zoom(self, f: float) -> None:
        self.scale = max(0.01, min(5.0, self.scale * f))
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        import time
        from forceview import draw_forces
        now = time.time()
        if not force and now - self._last < REFRESH_S:
            return
        self._last = now
        try:
            snap = self.sb.force_snapshot(self.hitch)
            draw_forces(self.canvas, snap, self.scale, self.w, self.h)
        except Exception:
            pass
