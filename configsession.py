"""The configuration being viewed in --view: which TOML files are loaded (and in what order), the manual edits on top, every
parameter with its value / source / whether changing it needs an XML (model) reload. No Tk, no MuJoCo: unit-testable.

Layers, lowest first (later wins): common.toml < the controller TOML < each enabled test file in order < manual edits.

PARAMETER LIST. The loader (config.py) reads some keys with a default written in the code, so they are not in any TOML. To
`python make_defaults.py` can write those hidden defaults into common.toml.

XML vs LIVE. A parameter "needs an XML reload" if changing it changes the generated MJCF. That is decided by experiment, not by a
hand-kept list: perturb the value, regenerate the XML, compare. Live parameters (controller gains, filters, planner...) are applied
to the running sandbox at once; XML ones are only recorded, flagged, and applied by the reload button.
"""
from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import config as _config
from config import _deep_merge, _flatten, _parse_set, config_from_raw, read_raw_config, resolve_test_path
from uischema import UISchema
from xml_builder import generate_mujoco_xml

BASE_DIR = Path(_config.__file__).resolve().parent
_MISSING = object()


class TrackedDict(dict):
    """dict that logs every key read through [] or .get(); a sub-table read with .get(k, {}) is tracked too."""

    def __init__(self, data, path, log):
        super().__init__(data)
        self._path, self._log = tuple(path), log

    def _wrap(self, key, val):
        if isinstance(val, dict) and not isinstance(val, TrackedDict):
            return TrackedDict(val, self._path + (key,), self._log)
        return val

    def __getitem__(self, key):
        val = self._wrap(key, super().__getitem__(key))
        if not isinstance(val, dict):
            self._log.setdefault(self._path + (key,), _MISSING)        # present in a TOML
        return val

    def get(self, key, default=None):
        if key in self:
            return self[key]
        val = self._wrap(key, default) if isinstance(default, dict) else default
        if not isinstance(val, dict):
            self._log.setdefault(self._path + (key,), default)         # hidden default in the code
        return val


def discover(raw: dict) -> dict:
    """{key path tuple: default (or _MISSING if the TOML has it)} for every key the loader reads, in the order it reads them."""
    log: dict = {}
    config_from_raw(TrackedDict(raw, (), log))
    return log


def dotted(path) -> str:
    return ".".join(path)


def get_path(raw: dict, path):
    node = raw
    for k in path:
        node = node[k]
    return node


def set_path(raw: dict, path, value) -> None:
    node = raw
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value


def _perturbations(path, value, raw):
    """Plausible different values for a parameter, used only to find out whether it changes the XML."""
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, max(value - 1, 0)]
    if isinstance(value, float):
        return [value * 1.25 + 0.01, value * 0.8 - 0.01]
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        return [[x * 1.25 + 0.01 for x in value], [x * 0.8 - 0.01 for x in value]]
    if path == ("controller", "active_path"):
        return [k for k in raw.get("paths", {}) if k != value][:1]
    return []


@dataclass
class ParamRow:
    path: tuple
    value: object          # current value (files + manual edits)
    loaded: object         # value from the files only (what "undo my edit" returns to)
    default: object        # default written in the code (or _MISSING when the key is in a TOML)
    source: str            # common / controller name / test file / "code default"
    manual: bool           # edited by hand
    kind: str              # "xml" | "live" | "unused"
    built: object = None   # value the running model was built with (xml rows)

    @property
    def key(self) -> str:
        return dotted(self.path)


class ConfigSession:
    def __init__(self, impl: str = "FixedDrawbarController", files=(), set_overrides=()):
        self.impl = impl
        self.enabled: list[str] = list(files)
        self.manual: dict[tuple, object] = {}
        for item in set_overrides:
            path, val = _parse_set(item)
            self.manual[tuple(path)] = val
        self.ui = UISchema.load()
        self._help: dict | None = None
        self._kinds: dict[tuple, str] | None = None
        self.built_raw: dict | None = None
        self.built_sig: str | None = None

    # -- files --------------------------------------------------------------------------------------------------------
    @staticmethod
    def available_files() -> list[tuple[str, str]]:
        """[(stem, one-line description)] for every configs/test/*.toml."""
        out = []
        for p in sorted((BASE_DIR / "configs" / "test").glob("*.toml")):
            name, desc = p.stem, ""
            try:
                raw = tomllib.loads(p.read_text())
                name_txt = str(raw.get("test", {}).get("TestName", ""))
                first = next((ln.lstrip("# ").strip() for ln in p.read_text().splitlines() if ln.startswith("#")), "")
                desc = (name_txt + ("  -  " + first if first else "")) if name_txt else first
            except Exception:
                pass
            out.append((name, desc))
        return out

    def set_enabled(self, name: str, on: bool) -> None:
        if on and name not in self.enabled:
            self.enabled.append(name)
        elif not on and name in self.enabled:
            self.enabled.remove(name)

    def move(self, name: str, delta: int) -> None:
        if name not in self.enabled:
            return
        i = self.enabled.index(name)
        j = max(0, min(len(self.enabled) - 1, i + delta))
        self.enabled.insert(j, self.enabled.pop(i))

    # -- raw layers ---------------------------------------------------------------------------------------------------
    def _impl_path(self) -> Path:
        p = Path(self.impl)
        if not p.suffix:
            p = BASE_DIR / "configs" / f"{p.name}.toml"
        return p if p.is_absolute() else BASE_DIR / p

    def _layers(self) -> list[tuple[str, dict]]:
        """[(label, flat {dotted: value})] for common, controller, then each enabled test file."""
        out = []
        for label, path in (("common", BASE_DIR / "configs" / "common.toml"), (Path(self.impl).stem, self._impl_path())):
            with open(path, "rb") as f:
                out.append((label, _flatten(tomllib.load(f))))
        for name in self.enabled:
            with open(resolve_test_path(name), "rb") as f:
                t = tomllib.load(f)
            t.pop("test", None)
            out.append((Path(name).stem, _flatten(t)))
        return out

    def raw_loaded(self) -> tuple[dict, list[dict]]:
        return read_raw_config(self._impl_path(), None, self.enabled)

    def raw_current(self) -> tuple[dict, list[dict]]:
        raw, infos = self.raw_loaded()
        for path, val in self.manual.items():
            set_path(raw, path, copy.deepcopy(val))
        return raw, infos

    def cfg(self):
        raw, infos = self.raw_current()
        return config_from_raw(raw, infos)

    def live_cfg(self):
        """Config for the RUNNING model: live parameters as edited, XML parameters held at the values the model was built with."""
        raw, infos = self.raw_current()
        if self.built_raw is not None:
            for path, kind in self.kinds().items():
                if kind == "xml":
                    try:
                        set_path(raw, path, copy.deepcopy(get_path(self.built_raw, path)))
                    except KeyError:
                        pass
        return config_from_raw(raw, infos)

    # -- parameters ---------------------------------------------------------------------------------------------------
    def kinds(self) -> dict[tuple, str]:
        """Classify every parameter as 'xml' or 'live' by regenerating the model with a perturbed value (cached)."""
        if self._kinds is not None:
            return self._kinds
        raw, infos = self.raw_current()
        base_xml = generate_mujoco_xml(config_from_raw(copy.deepcopy(raw), infos))
        kinds = {}
        for path, default in discover(copy.deepcopy(raw)).items():
            try:
                value = get_path(raw, path)
            except KeyError:
                value = default
            kind = "xml"
            tried = False
            for alt in _perturbations(path, value, raw):
                tried = True
                try:
                    r2 = copy.deepcopy(raw)
                    set_path(r2, path, alt)
                    xml2 = generate_mujoco_xml(config_from_raw(r2, infos))
                except Exception:
                    continue
                kind = "xml" if xml2 != base_xml else "live"
                break
            else:
                if not tried and isinstance(value, (list, tuple, dict)):
                    kind = "live"                     # waypoint lists etc.: not geometry
                if path[0] in ("paths", "controller", "planner", "gnss", "imu", "ir_camera", "loadcell", "viewer", "simulation") and not tried:
                    kind = "live"
            kinds[path] = kind
        self._kinds = kinds
        return kinds

    def params(self) -> list[ParamRow]:
        raw_l, _ = self.raw_loaded()
        raw_c, _ = self.raw_current()
        layers = self._layers()
        found = discover(copy.deepcopy(raw_c))
        kinds = self.kinds()
        rows = []
        flat_c = _flatten(raw_c)
        for path, default in found.items():
            key = dotted(path)
            src = "code default"
            for label, flat in layers:
                if key in flat:
                    src = label
            try:
                loaded = get_path(raw_l, path)
            except KeyError:
                loaded = default
            value = self.manual.get(path, loaded)
            built = None
            if self.built_raw is not None:
                try:
                    built = get_path(self.built_raw, path)
                except KeyError:
                    built = default
            rows.append(ParamRow(path, value, loaded, default, "manual" if path in self.manual else src,
                                 path in self.manual, kinds.get(path, "xml"), built))
        known = {dotted(p) for p in found}
        flat_l = _flatten(raw_l)
        for key, val in flat_c.items():
            if key in known or key.startswith("test."):
                continue
            path = tuple(key.split("."))
            if key.startswith("paths."):                # every course is listed (the controller only reads the active one)
                src = "code default"
                for label, flat in layers:
                    if key in flat:
                        src = label
                rows.append(ParamRow(path, val, flat_l.get(key, val), _MISSING, "manual" if path in self.manual else src,
                                     path in self.manual, "live", None))
            else:                                       # in a TOML but never read by the loader (usually a typo)
                rows.append(ParamRow(path, val, val, _MISSING, "unused", False, "unused", None))
        return rows

    def set_value(self, path, value) -> None:
        self.manual[tuple(path)] = value

    def clear_value(self, path) -> None:
        self.manual.pop(tuple(path), None)

    def set_linked(self, path, value) -> list[str]:
        """Set a parameter and the ones linked to it in ui.toml (e.g. controller class + drawbar mode); a value equal to what the
        files give is not kept as an edit. -> the dotted names that were set."""
        changes = {dotted(path): value}
        changes.update(self.ui.linked_changes(dotted(path), value))
        raw_l, _ = self.raw_loaded()
        for key, val in changes.items():
            p = tuple(key.split("."))
            try:
                loaded = get_path(raw_l, p)
            except KeyError:
                loaded = _MISSING
            if val == loaded:
                self.manual.pop(p, None)
            else:
                self.manual[p] = val
        return list(changes)

    def clear_linked(self, path) -> None:
        self.manual.pop(tuple(path), None)
        for link in self.ui.links:
            if dotted(path) in link.members:
                for m in link.members:
                    self.manual.pop(tuple(m.split(".")), None)

    def help_text(self, key: str) -> str:
        """The comment written beside the parameter in common.toml (or the controller file)."""
        if self._help is None:
            from make_readme import parse_toml_docs
            self._help = {}
            for p in (BASE_DIR / "configs" / "common.toml", self._impl_path()):
                try:
                    for (sec, k), (_, comment) in parse_toml_docs(p).items():
                        if comment:
                            self._help[f"{sec}.{k}"] = comment
                except Exception:
                    pass
        return self._help.get(key, "")

    def parse_text(self, text: str, like):
        """Parse the text typed in the window like a TOML value, of the same kind as `like` (so '1' stays a float)."""
        _, val = _parse_set(f"x.y={text}")
        if isinstance(like, bool):
            if not isinstance(val, bool):
                raise ValueError("expected true or false")
        elif isinstance(like, float):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError("expected a number")
            val = float(val)
        elif isinstance(like, int):
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError("expected a whole number")
        elif isinstance(like, str):
            val = str(val) if not isinstance(val, str) else val
        elif isinstance(like, (list, tuple)):
            if not isinstance(val, list):
                raise ValueError("expected a list like [0.8, 1.0]")
        return val

    # -- model (XML) state --------------------------------------------------------------------------------------------
    @staticmethod
    def xml_signature(cfg) -> str:
        import hashlib
        return hashlib.sha1(generate_mujoco_xml(cfg).encode()).hexdigest()

    def mark_built(self) -> None:
        """Call when the model has just been built from the current settings."""
        self.built_raw = copy.deepcopy(self.raw_current()[0])
        self.built_sig = self.xml_signature(self.cfg())

    def reload_needed(self) -> tuple[bool, list[str]]:
        """(model differs from the settings, [dotted names of XML parameters that differ from the built model])."""
        if self.built_raw is None:
            return False, []
        raw, _ = self.raw_current()
        names = []
        for path, kind in self.kinds().items():
            if kind != "xml":
                continue
            try:
                cur = get_path(raw, path)
            except KeyError:
                continue
            try:
                was = get_path(self.built_raw, path)
            except KeyError:
                was = cur
            if cur != was:
                names.append(dotted(path))
        return (self.xml_signature(self.cfg()) != self.built_sig), names

    # -- reporting ----------------------------------------------------------------------------------------------------
    def conflicts(self) -> dict[str, list[tuple[str, object]]]:
        """Parameters set to DIFFERENT values by more than one layer (common/controller count as one), last one wins."""
        seen: dict[str, list[tuple[str, object]]] = {}
        for label, flat in self._layers():
            for key, val in flat.items():
                seen.setdefault(key, []).append((label, val))
        out = {}
        for key, lst in seen.items():
            if len({json.dumps(v, default=str) for _, v in lst}) > 1 and len(lst) > 1:
                out[key] = lst
        return out

    def diff_vs_base(self) -> dict[tuple, object]:
        """Everything that differs from common + controller: the file layers and the manual edits."""
        base, _ = read_raw_config(self._impl_path(), None)
        cur, _ = self.raw_current()
        fb, fc = _flatten(base), _flatten(cur)
        out = {}
        for key, val in fc.items():
            if fb.get(key, _MISSING) != val:
                out[tuple(key.split("."))] = val
        return out

    def command_line(self) -> str:
        parts = []
        if self.enabled:
            parts.append("--tests <seconds>-" + ",".join(self.enabled))
        for path, val in self.manual.items():
            parts.append(f'--set {dotted(path)}={_toml_value(val)}')
        return " ".join(parts) or "(plain controller + common)"

    def save_toml(self, path: str | Path, title: str) -> Path:
        """Write the current differences (files + manual edits) as one test file."""
        diff = self.diff_vs_base()
        sections: dict[str, dict] = {}
        for p, v in diff.items():
            sections.setdefault(dotted(p[:-1]), {})[p[-1]] = v
        lines = ["# Saved from the --view window. Controller: " + Path(self.impl).stem,
                 "# Based on files: " + (", ".join(self.enabled) or "(none)"), "[test]", f"TestName = {json.dumps(title)}"]
        for sec, kv in sections.items():
            lines += ["", f"[{sec}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in kv.items()]
        path = Path(path)
        path.write_text("\n".join(lines) + "\n")
        return path


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(v, default=str)
