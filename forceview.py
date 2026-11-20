"""Live force diagram for one hitch (plan view, travelling to the right): drawn on any canvas with Tk's drawing methods.

The picture is the one in docs/hitch_forces.drawio, driven by live numbers: cable tensions (line colour/width), the hitch force N
(sum of the cable tensions plus the bar's axial force; tension pulls the trailers together), wheel thrust on both trailers, the
resulting force on the trailer behind (before rolling resistance), and a COMPRESSION banner when the cable sum falls below
its resting value by more than retension_thresh_n.
"""
from __future__ import annotations

CORNERS = ("tl", "tr", "bl", "br")


def make_snapshot(hitch, cable_n, bar_n, thrust_ahead_n, thrust_behind_n, baseline_n, thresh_n, yaw_rad=0.0):
    """Pure function: numbers in, the values the diagram shows out."""
    cs = float(sum(cable_n))
    tl, tr, bl, br = cable_n
    hitch_n = cs + (bar_n if bar_n is not None else 0.0)         # tension positive = pulling the two trailers together
    out = dict(
        hitch=hitch, cable_n=list(cable_n), bar_n=bar_n, cable_sum_n=cs, hitch_n=hitch_n,
        thrust_ahead_n=thrust_ahead_n, thrust_behind_n=thrust_behind_n,
        yaw_moment_n=(tl + bl) - (tr + br), pitch_moment_n=(tl + tr) - (bl + br),
        baseline_n=baseline_n, thresh_n=thresh_n, yaw_rad=yaw_rad,
        net_behind_n=thrust_behind_n + hitch_n,
    )
    # Compression = the hitch is NET pushed, or the cables have unloaded below their resting value.
    out["compression"] = (baseline_n is not None and cs < baseline_n - thresh_n) or (bar_n is not None and hitch_n < -thresh_n)
    # Self-stress: cables pull the trailers together while the bar pushes them apart, and the two cancel in N. A pre-tensioned
    # hitch has some by design (the resting cable sum); it is a FIGHT when the cables carry far more than that.
    out["self_stress_n"] = min(cs, -bar_n) if (bar_n is not None and bar_n < 0.0) else 0.0
    out["fight"] = (baseline_n is not None and bar_n is not None and bar_n < 0.0
                    and cs > baseline_n + max(thresh_n, 0.5 * baseline_n) and abs(hitch_n) < 0.25 * out["self_stress_n"])
    return out


def _clip(v, lim):
    return max(-lim, min(lim, v))


def _end(x0, length, width):
    """Arrow tip, kept inside the canvas."""
    return max(8.0, min(width - 8.0, x0 + length))


def draw_forces(canvas, snap, scale_px_per_n=0.25, width=720, height=420) -> None:
    c = canvas
    c.delete("all")
    if snap is None:
        c.create_text(width / 2, height / 2, text="(waiting for the simulation)", fill="#888888")
        return
    lim = 0.42 * width
    ax, ay = 0.74 * width, 0.5 * height          # trailer ahead (right) centre
    bx = 0.26 * width                            # trailer behind (left) centre
    th, tw = 0.30 * height, 0.22 * width
    c.create_text(10, 8, anchor="nw", fill="#333333", font=("Helvetica", 12, "bold"),
                  text=f"Hitch {snap['hitch']}  (plan view, travel →)   scale {scale_px_per_n * 100:.0f} px per 100 N")
    for cx, name in ((ax, "trailer j (ahead)"), (bx, "trailer j+1 (behind)")):
        c.create_rectangle(cx - tw / 2, ay - th / 2, cx + tw / 2, ay + th / 2, fill="#dbe9fb", outline="#2a7ab9", width=2)
        c.create_text(cx, ay - th / 2 + 12, text=name, fill="#2a7ab9")
    x1, x2 = bx + tw / 2, ax - tw / 2
    # bar and ball
    bar_col = "#444444" if (snap["bar_n"] is None or snap["bar_n"] >= 0) else "#d62728"
    c.create_line(x1, ay, x2, ay, width=8, fill=bar_col)
    c.create_oval((x1 + x2) / 2 - 12, ay - 12, (x1 + x2) / 2 + 12, ay + 12, fill="white", outline="#444444", width=2)
    # cables: top pair at ay - 0.4 th (left of travel), bottom pair at ay + 0.4 th; colour and width follow tension
    tmax = max(50.0, max(snap["cable_n"]))
    for k, (code, t) in enumerate(zip(CORNERS, snap["cable_n"])):
        yy = ay - 0.36 * th if code[1] == "l" else ay + 0.36 * th
        off = -6 if code[0] == "t" else 6                       # top and bottom cables drawn side by side
        yy += off
        w = 1 + 5 * t / tmax
        red = int(min(255, 90 + 165 * t / tmax))
        c.create_line(x1, yy, x2, yy, width=w, fill=f"#{red:02x}3030")
        label = f"{code} {t:.0f} N" if t >= 1.0 else f"{code} SLACK (0 N)"
        c.create_text((x1 + x2) / 2, yy + (-12 if off < 0 else 12), text=label, fill="#c0392b" if t >= 1.0 else "#888888", font=("Helvetica", 9))
    # hitch force N: on trailer j+1 it pulls forward (+x) when tension; on trailer j it pulls backward
    n = snap["hitch_n"]
    L = _clip(n * scale_px_per_n, lim)
    if abs(L) >= 2:
        c.create_line(x1, ay, _end(x1, L, width), ay, arrow="last", width=4, fill="#1f5fbf")
        c.create_line(x2, ay, _end(x2, -L, width), ay, arrow="last", width=4, fill="#1f5fbf")
    c.create_text((x1 + x2) / 2, ay - 0.5 * th - 18, fill="#1f5fbf", font=("Helvetica", 11, "bold"),
                  text=f"N = cables {snap['cable_sum_n']:.0f} + bar {snap['bar_n'] if snap['bar_n'] is not None else 0:.0f} = {n:+.0f} N")
    # wheel thrust and the force on the trailer behind
    for cx, f, lab in ((ax, snap["thrust_ahead_n"], "F_w"), (bx, snap["thrust_behind_n"], "F_w")):
        Lw = _clip(f * scale_px_per_n, lim)
        c.create_line(cx - 20, ay + th / 2 + 22, _end(cx - 20, Lw, width), ay + th / 2 + 22, arrow="last", width=5, fill="#2e8b57")
        c.create_text(cx - 20, ay + th / 2 + 38, anchor="w", fill="#2e8b57", text=f"{lab} wheel thrust {f:+.0f} N")
    Ln = _clip(snap["net_behind_n"] * scale_px_per_n, lim)
    c.create_line(bx, ay + th / 2 + 62, _end(bx, Ln, width), ay + th / 2 + 62, arrow="last", width=6, fill="#d62728")
    c.create_text(bx - tw / 2, ay + th / 2 + 78, anchor="w", fill="#d62728",
                  text=f"net on trailer j+1 = F_w + N = {snap['net_behind_n']:+.0f} N  (before rolling resistance)")
    c.create_text(10, height - 34, anchor="sw", fill="#333333",
                  text=f"yaw moment {snap['yaw_moment_n']:+.0f} N (left pair - right pair)    pitch moment {snap['pitch_moment_n']:+.0f} N (top - bottom)    hitch yaw {snap['yaw_rad'] * 57.2958:+.1f} deg")
    b = snap["baseline_n"]
    c.create_text(10, height - 14, anchor="sw", fill="#333333",
                  text=("cable sum baseline: learning..." if b is None else
                        f"cable sum {snap['cable_sum_n']:.0f} N vs resting {b:.0f} N (compression below {b - snap['thresh_n']:.0f} N)"))
    if snap["fight"]:
        c.create_rectangle(width * 0.12, 34, width * 0.88, 62, fill="#e67e22", outline="")
        c.create_text(width / 2, 48, fill="white", font=("Helvetica", 12, "bold"),
                      text=f"INTERNAL FIGHT: cables pull {snap['cable_sum_n']:.0f} N, bar pushes {-snap['bar_n']:.0f} N, they cancel in N")
    elif snap["compression"]:
        c.create_rectangle(width * 0.25, 34, width * 0.75, 62, fill="#d62728", outline="")
        c.create_text(width / 2, 48, fill="white", font=("Helvetica", 12, "bold"), text="COMPRESSION: hitch is being pushed")
