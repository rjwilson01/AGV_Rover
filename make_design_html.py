"""Renders design.xml as a readable page: docs/design.html (standard library only).

    python make_design_html.py [design.xml] [docs/design.html]

One table per section: value, unit, status badge, source, with the XML comments shown as notes next to the
element they explain. Unknown (tbd) items are listed at the top. Exits non-zero if the XML does not parse, and
warns about values that have no status (the status is inherited from the nearest parent that has one).
The output is generated: edit design.xml, not the HTML.
"""
from __future__ import annotations

import html
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUSES = ("design", "model", "assumed", "estimate", "derived", "tbd")
MEANING = {
    "design": "chosen or measured for the real build",
    "model": "value the simulator uses today; may differ from the build",
    "assumed": "placeholder to replace once the real part or number is known",
    "estimate": "first-order calculation or rule of thumb, not yet checked",
    "derived": "computed from other values or from a stated script",
    "tbd": "unknown, to be filled in",
}
META = {"status", "unit", "src"}       # attributes shown in their own columns


def parse(path: Path) -> ET.Element:
    """Parse keeping XML comments (ElementTree drops them by default)."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser).getroot()


def is_comment(e) -> bool:
    return e.tag is ET.Comment


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def label(e) -> str:
    ident = e.get("id") or e.get("cable") or e.get("ref")
    return f"{e.tag} [{ident}]" if ident else e.tag


def value_text(e) -> str:
    """Element text, else its non-meta attributes (so <cable_line id=.. lateral_sign=..> still shows something)."""
    t = clean(e.text)
    if t:
        return t
    attrs = [f"{k}={v}" for k, v in e.attrib.items() if k not in META and k not in ("id", "cable", "ref")]
    return ", ".join(attrs)


def walk(e, depth=0, inherited=None, out=None, todo=None, warn=None, path=""):
    """Flatten the tree into rows: ('note', depth, text) | ('head', depth, label) | ('item', depth, label, value, unit, status, src)."""
    out = [] if out is None else out
    todo = [] if todo is None else todo
    warn = [] if warn is None else warn
    for ch in e:
        if is_comment(ch):
            out.append(("note", depth, clean(ch.text)))
            continue
        status = ch.get("status") or inherited
        here = f"{path}/{label(ch)}" if path else label(ch)
        kids = [k for k in ch if not is_comment(k)]
        if kids:
            out.append(("head", depth, label(ch), ch.get("status"), ch.get("src"), value_text(ch)))
            walk(ch, depth + 1, status, out, todo, warn, here)
        else:
            val = value_text(ch)
            if status == "tbd":
                todo.append(here)
            elif val and status is None and ch.get("id") is None and ch.get("cable") is None and ch.get("ref") is None \
                    and not any(k not in META for k in ch.attrib):
                warn.append(f"{here}: value '{val}' has no status")
            out.append(("item", depth, label(ch), val, ch.get("unit") or "", status, ch.get("src") or ""))
    return out, todo, warn


def badge(status) -> str:
    if not status:
        return ""
    return f'<span class="b {html.escape(status)}">{html.escape(status)}</span>'


def render_rows(rows) -> str:
    parts = []
    for r in rows:
        kind, depth = r[0], r[1]
        pad = f' style="padding-left:{0.6 + 1.1 * depth:.1f}rem"'
        if kind == "note":
            if r[2]:
                parts.append(f'<tr class="note"><td colspan="5"{pad}>{html.escape(r[2])}</td></tr>')
        elif kind == "head":
            _, _, lab, status, src, val = r
            extra = f" {html.escape(val)}" if val else ""
            parts.append(f'<tr class="head"><td{pad}>{html.escape(lab)}<span class="muted">{extra}</span></td><td></td><td></td>'
                         f'<td>{badge(status)}</td><td class="src">{html.escape(src or "")}</td></tr>')
        else:
            _, _, lab, val, unit, status, src = r
            cls = " tbd" if status == "tbd" else ""
            shown = html.escape(val) if val else ('<span class="muted">TBD</span>' if status == "tbd" else "")
            parts.append(f'<tr class="item{cls}"><td{pad}>{html.escape(lab)}</td><td class="val">{shown}</td>'
                         f'<td class="unit">{html.escape(unit)}</td><td>{badge(status)}</td><td class="src">{html.escape(src)}</td></tr>')
    return "\n".join(parts)


CSS = """
:root{--bg:#fff;--fg:#1c1f23;--mut:#677;--line:#d8dde3;--head:#eef1f5;--note:#f7f8fa;--tbdbg:#fff4f2;
--design:#1d7a46;--model:#1f5fbf;--assumed:#a15c00;--estimate:#8a4fb0;--derived:#0b7a85;--tbd:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#15181c;--fg:#e6e9ed;--mut:#98a2ad;--line:#2d343c;--head:#20262d;--note:#1b2025;--tbdbg:#2a1a19;
--design:#5fcf8c;--model:#7fb0ff;--assumed:#f0b45a;--estimate:#c79bea;--derived:#5fd0da;--tbd:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:70rem;margin:0 auto;padding:1.5rem 16px 3rem}
h1{font-size:1.5rem;margin:.2rem 0}h2{font-size:1.1rem;margin:2rem 0 .5rem}
.sub{color:var(--mut);margin:0 0 1rem}.legend{display:flex;flex-wrap:wrap;gap:.4rem .9rem;margin:.6rem 0 1rem}
.legend span.m{color:var(--mut);font-size:.85rem}
.b{display:inline-block;padding:0 .5rem;border-radius:.7rem;border:1px solid currentColor;font-size:.78rem;font-weight:600}
.b.design{color:var(--design)}.b.model{color:var(--model)}.b.assumed{color:var(--assumed)}
.b.estimate{color:var(--estimate)}.b.derived{color:var(--derived)}.b.tbd{color:var(--tbd)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:.4rem}
table{border-collapse:collapse;width:100%;min-width:42rem}
td,th{padding:.28rem .6rem;border-top:1px solid var(--line);vertical-align:top;text-align:left}
th{background:var(--head);border-top:0;font-size:.85rem}
tr.head td{background:var(--head);font-weight:600}tr.note td{background:var(--note);color:var(--mut);font-size:.88rem}
tr.tbd td{background:var(--tbdbg)}.val{font-variant-numeric:tabular-nums;font-weight:600}
.muted{color:var(--mut);font-weight:400}.unit{color:var(--mut)}.src{color:var(--mut);font-size:.85rem}
ul.todo{columns:2;margin:.3rem 0;padding-left:1.2rem}@media(max-width:600px){ul.todo{columns:1}}
.gen{color:var(--mut);font-size:.82rem;margin-top:2rem}
"""


def build(root: ET.Element, src_name: str) -> tuple[str, list[str]]:
    sections, todo_all, warn_all, counts = [], [], [], Counter()
    intro = ""                      # a top-level comment just before a section describes that section
    for sec in root:
        if is_comment(sec):
            intro = clean(clean(sec.text).strip("= ").strip())
            continue
        rows, todo, warn = walk(sec, 0, sec.get("status"), None, None, None, label(sec))
        for r in rows:
            if r[0] == "item" and r[5]:
                counts[r[5]] += 1
        todo_all += todo
        warn_all += warn
        sections.append(("sec", label(sec), rows, intro))
        intro = ""
    legend = "".join(f'<span>{badge(s)} <span class="m">{html.escape(MEANING[s])}</span></span>' for s in STATUSES)
    tally = ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s])
    body = []
    for s in sections:
        desc = f'<p class="sub">{html.escape(s[3])}</p>' if s[3] else ""
        body.append(f'<h2>{html.escape(s[1])}</h2>{desc}<div class="wrap"><table><thead><tr><th>Item</th><th>Value</th><th>Unit</th>'
                    f'<th>Status</th><th>Source</th></tr></thead><tbody>\n{render_rows(s[2])}\n</tbody></table></div>')
    todo_html = ("<ul class=\"todo\">" + "".join(f"<li>{html.escape(t)}</li>" for t in todo_all) + "</ul>") if todo_all else "<p>None.</p>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGV Rover design</title><style>{CSS}</style></head>
<body><main>
<h1>AGV Rover design</h1>
<p class="sub">Schema version {html.escape(root.get('schema_version', '?'))}. {html.escape(tally)}.</p>
<div class="legend">{legend}</div>
<h2>To fill in ({len(todo_all)})</h2>
{todo_html}
{chr(10).join(body)}
<p class="gen">AUTO-GENERATED from {html.escape(src_name)} by make_design_html.py (make design). Edit the XML, not this page.</p>
</main></body></html>
"""
    return page, warn_all


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "design.xml"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "docs" / "design.html"
    try:
        root = parse(src)
    except ET.ParseError as ex:
        print(f"{src}: XML does not parse: {ex}", file=sys.stderr)
        return 1
    page, warn = build(root, src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(page, encoding="utf-8", newline="\n")
    for w in warn:
        print(f"warning: {w}", file=sys.stderr)
    print(f"{dst} written ({sum(1 for _ in root.iter())} elements)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
