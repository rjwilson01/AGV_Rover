"""Parse a test-sequence argument such as  200-a,b,c,100-d,e,100+f,g

Each group is  [seconds](-|+)file[,file...]
  -  reset to the default config, then load the group's files
  +  keep everything loaded so far, then load the group's files on top
The group runs once all its files are loaded, for `seconds` (if omitted: the last test_duration_s
given by an active test file, else DEFAULT_DURATION_S). Later files win over earlier ones.
"""
import re
from dataclasses import dataclass

DEFAULT_DURATION_S = 30.0
_GROUP = re.compile(r"^(?P<dur>\d+(?:\.\d+)?)?(?P<sign>[+-])(?P<file>.*)$")


@dataclass(frozen=True)
class Segment:
    reset: bool                      # True for '-', False for '+'
    duration_s: float | None         # from the argument; None = take it from the files / default
    new_files: tuple[str, ...]       # files introduced by this group
    active_files: tuple[str, ...]    # everything loaded when it runs (cumulative across '+')


def parse_sequence(spec: str) -> list[Segment]:
    spec = spec.strip()
    # tolerate a flag-style leading '-' as in "-200-a,b" (but not "-a", a reset group with file a)
    if re.match(r"^-\d+(?:\.\d+)?[+-]", spec):
        spec = spec[1:]
    groups: list[tuple[bool, float | None, list[str]]] = []
    for tok in (t.strip() for t in spec.split(",")):
        m = _GROUP.match(tok)
        if m:
            dur = float(m["dur"]) if m["dur"] else None
            groups.append((m["sign"] == "-", dur, [m["file"]] if m["file"] else []))
        elif tok:
            if not groups:
                groups.append((True, None, []))      # leading bare file: treat as a reset group
            groups[-1][2].append(tok)
    if not groups:
        raise ValueError("empty test sequence")
    segs, active = [], ()
    for reset, dur, files in groups:
        active = tuple(files) if reset else active + tuple(files)
        segs.append(Segment(reset, dur, tuple(files), active))
    return segs
