"""Reads configs/ui.toml: how the --view Parameters window groups parameters, which widget each gets, and which parameters are
linked. No Tk: unit-testable. Run `python uischema.py` to check the file against the real parameter list."""
from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

UI_PATH = Path(__file__).resolve().parent / "configs" / "ui.toml"
OTHER = "Other"


@dataclass
class Link:
    name: str
    members: list[str]
    rows: list[list]

    def row_for(self, key: str, value) -> list | None:
        i = self.members.index(key)
        return next((r for r in self.rows if r[i] == value), None)


@dataclass
class UISchema:
    groups: list[tuple[str, list[str]]] = field(default_factory=list)
    choices: dict[str, dict] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = UI_PATH) -> "UISchema":
        if not Path(path).exists():
            return cls()
        with open(path, "rb") as f:
            d = tomllib.load(f)
        return cls(
            groups=[(g["name"], list(g.get("match", []))) for g in d.get("group", [])],
            choices={k: dict(v) for k, v in d.get("choice", {}).items()},
            links=[Link(l["name"], list(l["members"]), [list(r) for r in l["rows"]]) for l in d.get("link", [])],
        )

    # -- grouping ---------------------------------------------------------------------------------------------------
    def group_of(self, key: str) -> str:
        for name, patterns in self.groups:
            if any(fnmatch.fnmatchcase(key, p) for p in patterns):
                return name
        return OTHER

    def group_order(self) -> list[str]:
        return [n for n, _ in self.groups] + [OTHER]

    # -- widgets ----------------------------------------------------------------------------------------------------
    def widget(self, key: str, value, raw: dict | None = None) -> tuple[str, list]:
        """-> ("bool"|"choice"|"text", options)."""
        if isinstance(value, bool):
            return "bool", [True, False]
        ch = self.choices.get(key)
        if ch is not None:
            if "options" in ch:
                opts = list(ch["options"])
            else:
                opts = list((raw or {}).get(ch.get("from", ""), {}))
            if isinstance(value, str) and value not in opts:
                opts.append(value)                       # never hide the current value (e.g. a path from an older file)
            if opts:
                return "choice", opts
        return "text", []

    # -- links ------------------------------------------------------------------------------------------------------
    def linked_changes(self, key: str, value) -> dict[str, object]:
        """Other parameters that must change when `key` is set to `value`."""
        out = {}
        for link in self.links:
            if key in link.members:
                row = link.row_for(key, value)
                if row is not None:
                    out.update({m: v for m, v in zip(link.members, row) if m != key})
        return out

    # -- checking ---------------------------------------------------------------------------------------------------
    def validate(self, keys: list[str]) -> list[str]:
        known = set(keys)
        problems = []
        for k in list(self.choices) + [m for l in self.links for m in l.members]:
            if k not in known:
                problems.append(f"ui.toml refers to {k}, which is not a parameter")
        for name, patterns in self.groups:
            for p in patterns:
                if not any(fnmatch.fnmatchcase(k, p) for k in known):
                    problems.append(f"ui.toml group '{name}': pattern {p} matches no parameter")
        for l in self.links:
            for r in l.rows:
                if len(r) != len(l.members):
                    problems.append(f"ui.toml link '{l.name}': row {r} does not match its members")
        other = [k for k in keys if self.group_of(k) == OTHER]
        if other:
            problems.append("not in any group (shown under 'Other'): " + ", ".join(other[:8]) + (" ..." if len(other) > 8 else ""))
        return problems


if __name__ == "__main__":
    from configsession import ConfigSession, dotted
    ses = ConfigSession("VariableDrawbarController", [])
    keys = [r.key for r in ses.params()]
    ui = UISchema.load()
    probs = ui.validate(keys)
    print(f"{len(keys)} parameters, {len(ui.groups)} groups, {len(ui.choices)} drop-downs, {len(ui.links)} links")
    for g in ui.group_order():
        n = sum(ui.group_of(k) == g for k in keys)
        print(f"  {g}: {n}")
    print("\n".join(probs) if probs else "ui.toml matches the parameter list")
