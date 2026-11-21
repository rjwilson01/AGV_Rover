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
    def __init__(self, ctl, tk_module=None, sandbox=None, session=None, on_change=None, show=(True, True, True)):
        if tk_module is None:
            import tkinter as tk_module            # may raise ImportError: the caller falls back to the viewer text panel
        self.tk = tk_module
        tk = self.tk
        self.ctl = ctl
        self.sandbox = sandbox
        self.session = session
        self.on_change = on_change or (lambda: None)
        self.reload_requested = False
        self.alive = True
        self.visible = True
        self._pending = set()          # window names asked for from the viewer's key callback (another thread): shown in update()
        self._after_id = None
        self._last_refresh = 0.0
        self.root = tk.Tk()
        self.root.title("AGV manual sandbox")
        self.root.protocol("WM_DELETE_WINDOW", self._close)       # closing only hides; F1..F4 in the viewer (or the boxes) bring it back
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

        # --- sub-windows and model reload
        wf = tk.LabelFrame(self.root, text="Windows")
        wf.pack(fill="x", padx=6, pady=2)
        self.sub = {}
        self.sub_vars = {}
        self.forces = self.configwin = self.paramwin = None
        self._want = dict(zip(("forces", "config", "params"), show))
        for key, label in (("forces", "Forces"), ("config", "Config files"), ("params", "Parameters")):
            var = tk.BooleanVar(value=bool(self._want[key]))
            self.sub_vars[key] = var
            tk.Checkbutton(wf, text=label, variable=var, command=lambda k=key: self._toggle_sub(k)).pack(side="left")
        self.pause_var = tk.BooleanVar(value=bool(ctl.pause_on_compression))
        tk.Checkbutton(wf, text="Pause on compression", variable=self.pause_var,
                       command=lambda: setattr(self.ctl, "pause_on_compression", bool(self.pause_var.get()))).pack(side="left", padx=8)
        tk.Button(wf, text="Show all", command=self.show_all).pack(side="left", padx=4)
        self.reload_btn = tk.Button(wf, text="Reload XML", command=self.request_reload)
        self.reload_btn.pack(side="right", padx=4)
        self.reload_lbl = tk.Label(wf, text="", fg="#b00000")
        self.reload_lbl.pack(side="right")

        # --- set-points
        self.sp = tk.LabelFrame(self.root, text="Set-points (click a row or use Left/Right; Up/Down or +/- change it, hold to repeat)")
        self.sp.pack(fill="x", padx=6, pady=4)
        self.rows = []
        self._build_setpoints()
        self._build_rest(sandbox)

    def _build_setpoints(self) -> None:
        tk, ctl, sp = self.tk, self.ctl, self.sp
        for w in sp.winfo_children():
            w.destroy()
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

    def _build_rest(self, sandbox) -> None:
        tk, ctl = self.tk, self.ctl
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
        if sandbox is not None:
            self.forces = ForceWindow(self.root, sandbox, tk)
        if self.session is not None:
            self.configwin = ConfigWindow(self.root, self.session, tk, self._changed)
            self.paramwin = ParamWindow(self.root, self.session, tk, self._changed, self.request_reload)
        self.sub = {"forces": self.forces, "config": self.configwin, "params": self.paramwin}
        for key, w in self.sub.items():
            if w is not None:
                w.set_visible(bool(self._want[key]))
        self.update_reload_state()
        self.refresh(force=True)

    # -- sub-windows / reload ------------------------------------------------------------------------------------------
    def _toggle_sub(self, key: str) -> None:
        w = self.sub.get(key)
        if w is not None:
            w.set_visible(bool(self.sub_vars[key].get()))

    def _changed(self) -> None:
        """A file was (de)selected or a parameter edited: apply what can be applied live, update the warnings."""
        self.on_change()
        self.update_reload_state()
        self.refresh(force=True)

    def update_reload_state(self) -> None:
        if self.session is None:
            return
        try:
            need, names = self.session.reload_needed()
        except Exception as exc:                                # a half-typed value must never kill the window
            self.reload_lbl.config(text=f"(cannot check: {exc})")
            return
        if need:
            shown = ", ".join(names[:3]) + (" ..." if len(names) > 3 else "")
            self.reload_lbl.config(text=f"MODEL OUT OF DATE ({len(names)} changed: {shown}) - press Reload XML")
            self.reload_btn.config(bg="#ffb3b3")
        else:
            self.reload_lbl.config(text="")
            self.reload_btn.config(bg=self.root.cget("bg"))
        for w in (self.configwin, self.paramwin):
            if w is not None:
                w.refresh(force=True)

    def request_reload(self) -> None:
        self.reload_requested = True

    # hide / show (nothing is ever destroyed, so a closed window can always be brought back)
    def set_main_visible(self, on: bool) -> None:
        self.visible = bool(on)
        (self.root.deiconify if on else self.root.withdraw)()
        if on:
            self.root.lift()

    def show_all(self) -> None:
        self.set_main_visible(True)
        for key, w in self.sub.items():
            if w is not None:
                w.set_visible(True)
                self.sub_vars[key].set(True)

    def request_show(self, name: str) -> None:
        """Thread-safe: ask for 'main', 'forces', 'config', 'params' or 'all' to be shown (handled on the next update())."""
        self._pending.add(name)

    def _handle_pending(self) -> None:
        while self._pending:
            name = self._pending.pop()
            if name in ("main", "all"):
                self.set_main_visible(True)
            if name == "all":
                self.show_all()
            w = self.sub.get(name)
            if w is not None:
                w.set_visible(True)
                self.sub_vars[name].set(True)

    def take_reload(self) -> bool:
        r, self.reload_requested = self.reload_requested, False
        return r

    def rebind(self, ctl, sandbox) -> None:
        """After an XML reload: new set-point list and a new simulation behind the same windows."""
        self.ctl, self.sandbox = ctl, sandbox
        self.pause_var.set(ctl.pause_on_compression)
        for name, var in self.flag_vars.items():
            var.set(ctl.flags[name])
        self._build_setpoints()
        if self.forces is not None:
            self.forces.rebind(sandbox)
        for w in (self.configwin, self.paramwin):
            if w is not None:
                w.refresh(force=True)
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
        self.set_main_visible(False)

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
        if self.pause_var.get() != self.ctl.pause_on_compression:
            self.pause_var.set(self.ctl.pause_on_compression)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", self.ctl.readout or "(readout appears once the simulation is running)")
        for key, w in self.sub.items():                   # keep the tick boxes in step with windows closed by their X
            if w is not None and bool(self.sub_vars[key].get()) != w.visible:
                self.sub_vars[key].set(w.visible)
        if self.forces is not None and self.forces.visible:
            self.forces.refresh()

    def update(self) -> bool:
        """Call every frame. False once the window has been closed."""
        if not self.alive:
            return False
        try:
            self._handle_pending()
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
        self.visible = True
        self.win.protocol("WM_DELETE_WINDOW", lambda: self.set_visible(False))
        self.bar = bar

    def set_visible(self, on: bool) -> None:
        self.visible = bool(on)
        (self.win.deiconify if on else self.win.withdraw)()

    def rebind(self, sandbox) -> None:
        self.sb = sandbox
        self.n_hitches = max(sandbox.cfg.num_trailers - 1, 1)
        self.hitch = min(self.hitch, self.n_hitches)

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


class _SubWindow:
    """Toplevel that hides instead of closing; the control window's check boxes show it again."""

    def _init_window(self, parent, tk, title: str, geometry: str):
        self.tk = tk
        self.visible = True
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.geometry(geometry)
        self.win.protocol("WM_DELETE_WINDOW", lambda: self.set_visible(False))

    def set_visible(self, on: bool) -> None:
        self.visible = bool(on)
        (self.win.deiconify if on else self.win.withdraw)()


class ConfigWindow(_SubWindow):
    """Which config files are loaded, and in what order (the LAST one wins where two set the same parameter)."""

    def __init__(self, parent, session, tk, on_change):
        self._init_window(parent, tk, "AGV config files", "760x620+40+40")
        self.session, self.on_change = session, on_change
        self.info = tk.Label(self.win, anchor="w", justify="left", wraplength=720)
        self.info.pack(fill="x", padx=6, pady=4)
        self.list_frame = tk.LabelFrame(self.win, text="Test files (tick to load; order = precedence, later wins)")
        self.list_frame.pack(fill="x", padx=6, pady=4)
        self.conflict_text = tk.Text(self.win, height=10, width=90, font=("Courier", 9))
        self.conflict_text.pack(fill="both", expand=True, padx=6, pady=4)
        bar = tk.Frame(self.win)
        bar.pack(fill="x", padx=6, pady=4)
        tk.Label(bar, text="Save current setup as configs/test/").pack(side="left")
        self.save_name = tk.StringVar(value="t_my_setup")
        tk.Entry(bar, textvariable=self.save_name, width=22).pack(side="left")
        tk.Label(bar, text=".toml").pack(side="left")
        tk.Button(bar, text="Save", command=self.save).pack(side="left", padx=6)
        tk.Button(bar, text="Clear my parameter edits", command=self.clear_edits).pack(side="right")
        self.status = tk.Label(self.win, anchor="w", fg="#005a00")
        self.status.pack(fill="x", padx=6)
        self._vars = {}
        self.refresh(force=True)

    def _toggle(self, name: str) -> None:
        self.session.set_enabled(name, bool(self._vars[name].get()))
        self.on_change()

    def _move(self, name: str, delta: int) -> None:
        self.session.move(name, delta)
        self.on_change()

    def clear_edits(self) -> None:
        self.session.manual.clear()
        self.on_change()

    def save(self) -> None:
        from pathlib import Path
        name = self.save_name.get().strip()
        if not name or any(c in name for c in "/\\ .:"):
            self.status.config(text="give a plain file name (letters, digits, _ or -)", fg="#b00000")
            return
        path = Path(__file__).resolve().parent / "configs" / "test" / f"{name}.toml"
        if path.exists():
            self.status.config(text=f"{path.name} already exists - choose another name", fg="#b00000")
            return
        try:
            self.session.save_toml(path, name.replace("_", " "))
            self.status.config(text=f"saved {path} (run it later with --tests <seconds>-{name})", fg="#005a00")
        except Exception as exc:
            self.status.config(text=f"could not save: {exc}", fg="#b00000")

    def refresh(self, force: bool = False) -> None:
        tk, ses = self.tk, self.session
        self.info.config(text=f"controller: {ses.impl}    equivalent command line: {ses.command_line()}")
        for w in self.list_frame.winfo_children():
            w.destroy()
        files = dict(ses.available_files())
        order = list(ses.enabled) + [n for n in files if n not in ses.enabled]
        self._vars = {}
        for i, name in enumerate(order):
            on = name in ses.enabled
            var = tk.BooleanVar(value=on)
            self._vars[name] = var
            tk.Checkbutton(self.list_frame, variable=var, command=lambda n=name: self._toggle(n)).grid(row=i, column=0)
            tk.Label(self.list_frame, text=f"{ses.enabled.index(name) + 1}." if on else "", width=3).grid(row=i, column=1)
            if on:
                tk.Button(self.list_frame, text="^", width=2, command=lambda n=name: self._move(n, -1)).grid(row=i, column=2)
                tk.Button(self.list_frame, text="v", width=2, command=lambda n=name: self._move(n, +1)).grid(row=i, column=3)
            tk.Label(self.list_frame, text=f"{name}   {files.get(name, '')}"[:100], anchor="w",
                     font=("Courier", 9, "bold" if on else "normal")).grid(row=i, column=4, sticky="w")
        lines = []
        conflicts = ses.conflicts()
        if conflicts:
            lines.append("Parameters set by more than one file (the last one listed wins):")
            for key, lst in conflicts.items():
                lines.append(f"  {key}: " + "  <  ".join(f"{lab}={val}" for lab, val in lst))
        else:
            lines.append("No parameter is set to different values by different files.")
        if ses.manual:
            lines.append("")
            lines.append("Your edits from the Parameters window (applied on top of every file):")
            lines += [f"  {'.'.join(p)} = {v}" for p, v in ses.manual.items()]
        self.conflict_text.delete("1.0", "end")
        self.conflict_text.insert("1.0", "\n".join(lines))


class _Scroll:
    """Vertically scrollable frame (Canvas + inner Frame)."""

    def __init__(self, parent, tk):
        self.outer = tk.Frame(parent)
        self.canvas = tk.Canvas(self.outer, highlightthickness=0)
        sb = tk.Scrollbar(self.outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))
        self.inner.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))


class ParamWindow(_SubWindow):
    """Every parameter, grouped as configs/ui.toml says, in two lists: LIVE (applied at once) and XML RELOAD (applied by the
    Reload XML button). Tick boxes for booleans, drop-downs for the parameters ui.toml gives choices for, text boxes otherwise."""

    def __init__(self, parent, session, tk, on_change, on_reload):
        self._init_window(parent, tk, "AGV parameters", "1380x720+60+60")
        self.session, self.on_change, self.on_reload = session, on_change, on_reload
        top = tk.Frame(self.win)
        top.pack(fill="x", padx=6, pady=4)
        tk.Label(top, text="Filter:").pack(side="left")
        self.filter = tk.StringVar(value="")
        e = tk.Entry(top, textvariable=self.filter, width=24)
        e.pack(side="left")
        self.filter.trace_add("write", lambda *a: self._apply_filter())
        tk.Button(top, text="Reload XML", command=on_reload).pack(side="right")
        self.banner = tk.Label(top, text="", fg="#b00000", anchor="e")
        self.banner.pack(side="right", padx=8)
        body = tk.Frame(self.win)
        body.pack(fill="both", expand=True, padx=6)
        self.panes = {}
        for col, (kind, title) in enumerate((("live", "LIVE parameters - take effect immediately"),
                                             ("xml", "XML-RELOAD parameters - change the model; press Reload XML"))):
            frame = tk.LabelFrame(body, text=title)
            frame.grid(row=0, column=col, sticky="nsew", padx=2)
            body.columnconfigure(col, weight=1)
            sc = _Scroll(frame, tk)
            sc.outer.pack(fill="both", expand=True)
            self.panes[kind] = sc
        body.rowconfigure(0, weight=1)
        self.status = tk.Label(self.win, anchor="w", justify="left", wraplength=1300)
        self.status.pack(fill="x", padx=6)
        self.rows = {}            # key -> dict(var, ent, lab, src, undo, widgets, wtype)
        self.headers = []         # [(header widget, [keys of its rows])]
        self._built = None
        self.refresh(force=True)

    # -- building / refreshing --------------------------------------------------------------------------------------
    def _build(self, rows, raw) -> None:
        tk, ui = self.tk, self.session.ui
        for sc in self.panes.values():
            for w in sc.inner.winfo_children():
                w.destroy()
        self.rows, self.headers = {}, []
        order = {g: n for n, g in enumerate(ui.group_order())}
        unused = "Unused by the code (typo?)"

        def group(r):
            return unused if r.kind == "unused" else ui.group_of(r.key)

        ordered = sorted(rows, key=lambda r: 999 if r.kind == "unused" else order[group(r)])       # stable inside a group
        n = {"live": 0, "xml": 0}
        last = {"live": None, "xml": None}
        members = None
        for r in ordered:
            kind = "xml" if r.kind == "xml" else "live"
            sc = self.panes[kind]
            if last[kind] != group(r):
                last[kind] = group(r)
                hdr = tk.Label(sc.inner, text=group(r), anchor="w", bg="#dde6f0", font=("TkDefaultFont", 9, "bold"))
                hdr.grid(row=n[kind], column=0, columnspan=4, sticky="we", pady=(6, 0))
                n[kind] += 1
                members = []
                self.headers.append((hdr, members))
            i = n[kind]
            n[kind] += 1
            members.append(r.key)
            wtype, opts = ui.widget(r.key, r.value, raw) if r.kind != "unused" else ("text", [])
            lab = tk.Label(sc.inner, text=r.key, anchor="w", width=34)
            if wtype == "bool":
                var = tk.BooleanVar(value=bool(r.value))
                ent = tk.Checkbutton(sc.inner, variable=var, anchor="w", command=lambda k=r.path: self._commit(k))
            elif wtype == "choice":
                var = tk.StringVar(value=str(r.value))
                ent = tk.OptionMenu(sc.inner, var, *[str(o) for o in opts], command=lambda v, k=r.path: self._commit(k))
                ent.config(width=24, anchor="w")
            else:
                var = tk.StringVar(value=_fmt(r.value))
                ent = tk.Entry(sc.inner, textvariable=var, width=26)
                ent.bind("<Return>", lambda e, k=r.path: self._commit(k))
                ent.bind("<FocusOut>", lambda e, k=r.path: self._commit(k))
                ent.bind("<FocusIn>", lambda e, k=r.key: self._help(k))
            src = tk.Label(sc.inner, text=r.source, anchor="w", width=18, fg="#555555")
            undo = tk.Button(sc.inner, text="undo", width=4, command=lambda k=r.path: self._undo(k))
            lab.grid(row=i, column=0, sticky="w")
            ent.grid(row=i, column=1, sticky="w")
            src.grid(row=i, column=2, sticky="w")
            undo.grid(row=i, column=3)
            lab.bind("<Enter>", lambda e, k=r.key: self._help(k))
            self.rows[r.key] = dict(var=var, ent=ent, lab=lab, src=src, undo=undo, wtype=wtype, widgets=(lab, ent, src, undo))
        self._built = self._signature(rows, raw)

    def _signature(self, rows, raw):
        ui = self.session.ui
        return (tuple(r.key for r in rows), tuple((r.key, tuple(map(str, ui.widget(r.key, r.value, raw)[1]))) for r in rows
                                                 if r.kind != "unused" and ui.widget(r.key, r.value, raw)[0] == "choice"))

    def refresh(self, force: bool = False) -> None:
        try:
            rows = self.session.params()
            raw, _ = self.session.raw_current()
        except Exception as exc:
            self.status.config(text=f"cannot list parameters: {exc}", fg="#b00000")
            return
        if self._built != self._signature(rows, raw):
            self._build(rows, raw)
        focus = None
        try:
            focus = self.win.focus_get()
        except Exception:
            pass
        for r in rows:
            w = self.rows.get(r.key)
            if w is None:
                continue
            if focus is not w["ent"]:
                w["var"].set(bool(r.value) if w["wtype"] == "bool" else (str(r.value) if w["wtype"] == "choice" else _fmt(r.value)))
            w["src"].config(text=r.source)
            differs_built = r.kind == "xml" and r.built is not None and r.value != r.built
            w["ent"].config(bg="#ffb36b" if differs_built else ("#fff2a8" if r.manual else "white"))
        need, names = self.session.reload_needed()
        self.banner.config(text=(f"MODEL OUT OF DATE: {len(names)} XML parameter(s) differ from the loaded model - press Reload XML"
                                 if need else "model matches the settings"))
        self._apply_filter()

    def _apply_filter(self) -> None:
        text = self.filter.get().strip().lower()
        shown = {}
        for key, w in self.rows.items():
            show = not text or text in key.lower()
            shown[key] = show
            for widget in w["widgets"]:
                (widget.grid if show else widget.grid_remove)()
        for hdr, keys in self.headers:
            (hdr.grid if any(shown.get(k, False) for k in keys) else hdr.grid_remove)()

    def _help(self, key: str) -> None:
        txt = self.session.help_text(key)
        self.status.config(text=f"{key}: {txt}" if txt else key, fg="#333333")

    # -- editing -----------------------------------------------------------------------------------------------------
    def _row(self, path):
        return next((r for r in self.session.params() if r.path == tuple(path)), None)

    def _commit(self, path) -> None:
        key = ".".join(path)
        w = self.rows.get(key)
        r = self._row(path)
        if w is None or r is None:
            return
        if w["wtype"] == "bool":
            val = bool(w["var"].get())
        elif w["wtype"] == "choice":
            val = w["var"].get()
        else:
            text = w["var"].get().strip()
            if text == _fmt(r.value):
                return
            try:
                val = self.session.parse_text(text, r.loaded if r.loaded is not _MISSING_ else r.default)
            except Exception as exc:
                w["ent"].config(bg="#ff9a9a")
                self.status.config(text=f"{key}: {exc}", fg="#b00000")
                return
        if val == r.value:
            return
        changed = self.session.set_linked(path, val)
        also = [k for k in changed if k != key]
        self.status.config(text=f"{key} = {_fmt(val)}" + (f"   (also set {', '.join(also)})" if also else "")
                           + ("   (XML parameter: press Reload XML)" if r.kind == "xml" else "   (applied)"),
                           fg="#b00000" if r.kind == "xml" else "#005a00")
        self.on_change()

    def _undo(self, path) -> None:
        self.session.clear_linked(path)
        self.on_change()


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


try:
    from configsession import _MISSING as _MISSING_
except Exception:                                  # keep the module importable without the config code
    _MISSING_ = object()
