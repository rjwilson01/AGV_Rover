"""Pick a MuJoCo model from the repo in a dropdown and open it in the MuJoCo viewer.

Run from anywhere:   python Subdesigns/model_picker.py            (searches the repo root)
                     python Subdesigns/model_picker.py --root C:\\path\\to\\folder

It lists every *.xml under the root whose contents contain a <mujoco> element (so pj_auto_layout.xml and the like are
skipped; *.xml.j2 templates are not listed because they need rendering first, use drawbar_and_balljoint.py for those).
Selecting a model compiles it straight away and shows either a one-line summary or the exact MuJoCo error, so a bad path or
bad attribute is reported here instead of failing in the viewer. "Load" then opens the viewer in its own process, so you can
keep several models open at once and closing one does not touch the others.
"""
import argparse
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env", "build", "dist"}
DEFAULT_ROOT = Path(__file__).resolve().parent.parent       # repo root (this file lives in Subdesigns/)


def is_mjcf(path: Path) -> bool:
    """True if the file looks like a MuJoCo model: a <mujoco ...> element near the top, even if the rest is broken."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(8192)
    except OSError:
        return False
    return "<mujoco" in head


def find_models(root: Path) -> list[Path]:
    """All MuJoCo model files under root, sorted by relative path."""
    found = []
    for p in root.rglob("*.xml"):
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel_parts[:-1]):
            continue
        if is_mjcf(p):
            found.append(p)
    return sorted(found, key=lambda q: str(q.relative_to(root)).lower())


def compile_summary(path: Path) -> tuple[bool, str]:
    """Compile the model. Returns (ok, one-line summary or the MuJoCo error text)."""
    try:
        import mujoco
    except ImportError:
        return False, "The 'mujoco' Python package is not installed in this interpreter (pip install mujoco)."
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:                                        # mujoco raises ValueError with the compiler message
        return False, str(e)
    return True, (f"OK   bodies {m.nbody - 1}   joints {m.njnt}   geoms {m.ngeom}   actuators {m.nu}   "
                  f"tendons {m.ntendon}   sensors {m.nsensor}   dt {m.opt.timestep:g} s")


class ModelPicker(tk.Tk):
    def __init__(self, root_dir: Path):
        super().__init__()
        self.title("MuJoCo model picker")
        self.minsize(640, 260)
        self.root_dir = root_dir
        self.models: list[Path] = []
        self.procs: list[subprocess.Popen] = []

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, text=f"Models under {root_dir}").grid(row=0, column=0, columnspan=3, sticky="w")

        self.choice = tk.StringVar()
        self.combo = ttk.Combobox(frm, textvariable=self.choice, state="readonly", width=70)
        self.combo.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self.on_select())
        ttk.Button(frm, text="Refresh", command=self.refresh).grid(row=1, column=1, padx=(8, 0))
        self.load_btn = ttk.Button(frm, text="Load", command=self.load, state="disabled")
        self.load_btn.grid(row=1, column=2, padx=(8, 0))

        self.status = tk.Text(frm, height=8, wrap="word", state="disabled", relief="flat",
                              background=self.cget("background"))
        self.status.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        frm.rowconfigure(2, weight=1)
        self.status.tag_configure("ok", foreground="#1a7f37")
        self.status.tag_configure("err", foreground="#b42318")
        self.status.tag_configure("info", foreground="#555555")

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.refresh()

    # -- helpers ---------------------------------------------------------------------------------------------------
    def say(self, text: str, tag: str = "info") -> None:
        self.status.configure(state="normal")
        self.status.delete("1.0", "end")
        self.status.insert("end", text, tag)
        self.status.configure(state="disabled")

    def selected(self) -> Path | None:
        i = self.combo.current()
        return self.models[i] if 0 <= i < len(self.models) else None

    # -- actions ---------------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        keep = self.selected()
        self.models = find_models(self.root_dir)
        self.combo["values"] = [str(p.relative_to(self.root_dir)) for p in self.models]
        if not self.models:
            self.choice.set("")
            self.load_btn.configure(state="disabled")
            self.say("No MuJoCo model files (*.xml containing <mujoco>) found under this folder.", "err")
            return
        if keep in self.models:
            self.combo.current(self.models.index(keep))
            self.on_select()
        else:
            self.choice.set("")
            self.load_btn.configure(state="disabled")
            self.say(f"{len(self.models)} model(s) found. Pick one from the list.", "info")

    def on_select(self) -> None:
        path = self.selected()
        if path is None:
            return
        ok, msg = compile_summary(path)
        self.say(msg, "ok" if ok else "err")
        self.load_btn.configure(state="normal" if ok else "disabled")

    def load(self) -> None:
        path = self.selected()
        if path is None:
            return
        try:
            proc = subprocess.Popen([sys.executable, "-m", "mujoco.viewer", f"--mjcf={path}"], cwd=str(path.parent))
        except OSError as e:
            self.say(f"Could not start the viewer: {e}", "err")
            return
        self.procs = [p for p in self.procs if p.poll() is None] + [proc]
        self.say(f"Opened {path.relative_to(self.root_dir)} in the MuJoCo viewer "
                 f"({len(self.procs)} viewer window(s) open from here).", "ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder to search (default: the repo root)")
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    ModelPicker(root).mainloop()


if __name__ == "__main__":
    main()
