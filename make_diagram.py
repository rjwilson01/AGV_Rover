"""Generate docs/architecture.drawio (open it at https://app.diagrams.net or in the draw.io desktop app).

Run `python make_diagram.py` after changing the control structure so the diagram stays in step with the code."""
from pathlib import Path
from xml.sax.saxutils import escape

cells, edges = [], []
_id = [1]


def box(x, y, w, h, text, fill="#ffffff", stroke="#666666", bold=False, rounded=True, font=11):
    _id[0] += 1
    cid = f"n{_id[0]}"
    style = (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
             f"fontSize={font};align=left;spacingLeft=6;verticalAlign=top;" + ("fontStyle=1;" if bold else ""))
    cells.append(f'<mxCell id="{cid}" value="{escape(text).replace(chr(10), "&lt;br&gt;")}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def lane(x, y, w, h, title, fill):
    _id[0] += 1
    cid = f"n{_id[0]}"
    cells.append(f'<mxCell id="{cid}" value="{escape(title)}" style="swimlane;horizontal=1;startSize=26;fillColor={fill};strokeColor=#999999;'
                 f'fontStyle=1;fontSize=13;rounded=0;" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def arrow(a, b, label="", dashed=False, color="#444444"):
    _id[0] += 1
    style = f"endArrow=block;html=1;strokeColor={color};fontSize=10;" + ("dashed=1;" if dashed else "")
    edges.append(f'<mxCell id="e{_id[0]}" value="{escape(label)}" style="{style}" edge="1" parent="1" source="{a}" target="{b}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')


# ---- lanes
lane(20, 20, 250, 560, "1  Inputs", "#f5f5f5")
lane(290, 20, 330, 560, "2  Lead steering (path following)", "#e1f0ff")
lane(640, 20, 330, 560, "3  What the followers are told", "#e8f6e1")
lane(990, 20, 360, 560, "4  Hitch laws (per hitch j)", "#fff2d6")
lane(1370, 20, 350, 560, "5  Wheel laws", "#fde4e4")
lane(1740, 20, 330, 560, "6  Actuators (MuJoCo)", "#ece4f7")

# ---- 1 inputs
cfgb = box(35, 60, 220, 120, "Config layers\ncommon.toml < <Preset>.toml < test files < --set\n(config.py)", "#ffffff")
wp = box(35, 200, 220, 90, "Waypoints (active_path)\ne.g. u_turn_wide", "#ffffff")
ir = box(35, 320, 220, 100, "IR camera (per hitch)\nrelative yaw / pitch / roll of trailer j+1 in trailer j\n(ir_camera.py)", "#ffffff", "#2a7ab9")
gn = box(35, 440, 220, 60, "GNSS on the lead -> true speed\n(gnss.py)  |  IMU per trailer: logged only", "#ffffff", "#2a7ab9")

# ---- 2 lead
pl = box(305, 60, 300, 100, "Planner (planner.py)\nrounds corners so the curvature never exceeds what the hitches allow\n(planner_margin, hitch yaw range); straight run-out past the goal", "#ffffff")
pp = box(305, 180, 300, 100, "Pure pursuit\nlookahead = max(lookahead_dist, gain x v); steer to the point ahead on the path\nkappa_raw = 2 y / L^2", "#ffffff")
sh = box(305, 300, 300, 90, "Curvature shaping\nlow-pass + slew limit, in DISTANCE (curv_filter_m, curv_rate_limit_per_m)\n-> kappa_lead", "#ffffff")
lv = box(305, 410, 300, 90, "Lead wheel command\nspeed v_cmd (v_center, start ramp)\nomega_lead = v_cmd x kappa_lead", "#ffffff")

# ---- 3 followers
rf = box(655, 60, 300, 110, "Ideal path reference\ns_ref = lead's monotone, rate-limited progress along the planned path\nkappa_ref(s_ref - delay)   (ref_smooth_m smoothing)", "#ffffff")
dv = box(655, 190, 300, 100, "Lead deviation\n(kappa_lead - kappa_ref) low-passed 2 m, clipped\nso a lead that corrects an error does not drag the chain\n(ref_dev_gain)", "#ffffff")
kf = box(655, 310, 300, 90, "kappa_follow(j) = kappa_ref + deviation\nread at the delay distance of hitch j", "#d6f0c8", bold=True)
sg = box(655, 420, 300, 120, "Stall guard (safeguard.py)\nlead commanded vs GNSS speed: if the lead stalls the followers are scaled down so they do not shove it", "#ffffff")

# ---- 4 hitch laws
tl = box(1005, 60, 330, 120, "Tightness / bar length law (Variable build)\ntightness = |yaw needed at full length| / yaw range\nr = r_max - tightness (r_max - r_min), same for front f\nthen shortened further if the hitch pitch would ground the bars (bar_pitch_limit)", "#ffffff")
ty = box(1005, 200, 330, 110, "Tube yaw law (feed-forward)\npsi_r = g atan(r kappa_follow), psi_f = g atan(f kappa_follow)\neach clipped to its tube range; total clipped to the hitch yaw range\nthe ball takes the rest", "#ffffff")
ca = box(1005, 330, 330, 150, "Cable length (main.py / bar_geometry.py)\nfor each of 4 corners: |anchor distance| from\npsi_r, psi_f (MEASURED), ball yaw = cmd - tubes, ball pitch/roll\n(held from IR), bar lengths (MEASURED)\n- preload offset", "#ffffff")
nofb = box(1005, 500, 330, 70, "NOTE: bar length and tube yaw are feed-forward on path curvature. There is no feedback on the measured hitch angle here.", "#fff8e1", "#d6a000")

# ---- 5 wheels
wl = box(1385, 60, 320, 200, "Follower wheel law (BaseSnakeController._ir_wheel_vels)\nstate (v_j, w_j) of trailer j; psi = MEASURED IR hitch yaw\nball position b in trailer j, a in trailer j+1 from the tube yaws and bar lengths\nball velocity matches on both sides, no lateral slip of trailer j+1\n-> (v_{j+1}, w_{j+1});  floor 0.3 v, clip +-3 rad/s", "#ffffff")
mp = box(1385, 280, 320, 110, "Wheel map\nleft = (v - w half_track)/r\nright = (v + w half_track)/r\nleft turn -> RIGHT wheel faster\nfollowers x guard scale, all x start ramp", "#fbd5d5", bold=True)

# ---- 6 actuators
aw = box(1755, 60, 300, 80, "Wheel motors (velocity actuators)\nm_t{i}_l, m_t{i}_r", "#ffffff")
asl = box(1755, 160, 300, 80, "Telescoping bars (position actuators)\nact_slide_rear_j, act_slide_front_j+1", "#ffffff")
ayw = box(1755, 260, 300, 80, "Tube yaw hinges (position actuators)\nact_yaw_rear_j, act_yaw_front_j+1", "#ffffff")
acp = box(1755, 360, 300, 80, "Cable capstans (tendon length, pull only)\ncap_t{j}_t{j+1}_tl/tr/bl/br (4 per hitch)", "#ffffff")
plant = box(1755, 460, 300, 90, "MuJoCo plant: trailers, wheels, passive ball joints, cables as tendons, ground contact", "#d9ccf0", bold=True)

# ---- arrows
arrow(cfgb, pl, "", True); arrow(wp, pl); arrow(pl, pp); arrow(pp, sh); arrow(sh, lv)
arrow(pl, rf, "planned path"); arrow(sh, dv, "kappa_lead"); arrow(rf, dv, "kappa_ref", True)
arrow(rf, kf); arrow(dv, kf)
arrow(kf, tl); arrow(kf, ty); arrow(tl, ty, "lengths", True)
arrow(ir, wl, "measured psi"); arrow(ir, tl, "pitch", True); arrow(ir, ca, "ball pitch/roll", True)
arrow(tl, wl, "bar lengths"); arrow(ty, wl, "tube yaws"); arrow(ty, ca); arrow(tl, ca)
arrow(lv, wl, "v, w of the lead"); arrow(wl, mp); arrow(gn, sg, "true speed"); arrow(lv, sg, "commanded speed"); arrow(sg, mp, "scale")
arrow(mp, aw); arrow(tl, asl); arrow(ty, ayw); arrow(ca, acp)
for a in (aw, asl, ayw, acp):
    arrow(a, plant)
arrow(plant, ir, "sensed", True, "#2a7ab9")

xml = ('<mxfile host="app.diagrams.net"><diagram name="AGV control structure" id="agv1"><mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" '
       'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2100" pageHeight="640" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "".join(edges) + '</root></mxGraphModel></diagram></mxfile>')
Path("docs/architecture.drawio").write_text(xml, encoding="utf-8")
print("wrote docs/architecture.drawio:", len(cells), "cells,", len(edges), "arrows")


# ======================================================================================================================
# Second diagram: forces on ONE hitch (plan view) - the "stability, not push/pull" picture
# ======================================================================================================================
def _forces_diagram():
    cs, es = [], []
    n = [100]

    def rect(x, y, w, h, text, fill="#ffffff", stroke="#666666", font=11, align="center"):
        n[0] += 1
        cs.append(f'<mxCell id="f{n[0]}" value="{escape(text).replace(chr(10), "&lt;br&gt;")}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};'
                  f'strokeColor={stroke};fontSize={font};align={align};verticalAlign=middle;" vertex="1" parent="1">'
                  f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def text(x, y, w, h, t, color="#333333", font=11, bold=False):
        n[0] += 1
        cs.append(f'<mxCell id="f{n[0]}" value="{escape(t).replace(chr(10), "&lt;br&gt;")}" style="text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;'
                  f'fontSize={font};fontColor={color};' + ("fontStyle=1;" if bold else "") + f'" vertex="1" parent="1">'
                  f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def vec(x1, y1, x2, y2, color, width=3, dashed=False, label=""):
        n[0] += 1
        st = f"endArrow=block;html=1;strokeColor={color};strokeWidth={width};fontSize=10;fontColor={color};" + ("dashed=1;" if dashed else "")
        es.append(f'<mxCell id="f{n[0]}" value="{escape(label)}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                  f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')

    def line(x1, y1, x2, y2, color, width=2, dashed=False):
        n[0] += 1
        st = f"endArrow=none;html=1;strokeColor={color};strokeWidth={width};" + ("dashed=1;" if dashed else "")
        es.append(f'<mxCell id="f{n[0]}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                  f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')

    text(40, 10, 1100, 30, "Forces on one hitch (plan view, travelling to the RIGHT): the hitch STABILISES, the wheels STEER", bold=True, font=15)
    # trailers: j ahead (right), j+1 behind (left)
    rect(60, 200, 220, 130, "trailer j+1 (behind)", "#dbe9fb", "#2a7ab9")
    rect(520, 200, 220, 130, "trailer j (ahead)", "#dbe9fb", "#2a7ab9")
    # bar + ball
    line(280, 265, 520, 265, "#444444", 8)
    rect(385, 250, 30, 30, "", "#ffffff", "#444444")
    text(330, 285, 140, 20, "bar + ball joint", font=10)
    # four cables (two shown per side in plan view: top and bottom cables overlap)
    for y1, y2 in ((215, 215), (315, 315)):
        line(280, y1, 520, y2, "#c0392b", 2)
    text(330, 190, 220, 20, "cables: T_tl+T_bl (left pair)", "#c0392b", 10)
    text(330, 318, 220, 20, "cables: T_tr+T_br (right pair)", "#c0392b", 10)
    # hitch axial force N (equal and opposite on the two trailers)
    vec(280, 265, 340, 265, "#1f5fbf", 3)
    vec(520, 265, 460, 265, "#1f5fbf", 3)
    text(345, 232, 110, 20, "N (+ tension)", "#1f5fbf", 10, True)
    # wheel thrust and rolling resistance
    vec(160, 345, 260, 345, "#2e8b57", 4); text(110, 350, 200, 20, "F_w: wheel thrust (wheel law)", "#2e8b57", 10, True)
    vec(620, 345, 720, 345, "#2e8b57", 4); text(570, 350, 200, 20, "F_w: wheel thrust", "#2e8b57", 10, True)
    vec(100, 185, 40, 185, "#888888", 2); text(40, 160, 160, 20, "R: rolling resistance", "#888888", 10)
    # net residual and desired direction
    vec(170, 265, 230, 265, "#d62728", 5, label="")
    text(150, 130, 260, 36, "net = F_w - R + N  (residual, accelerates the trailer)", "#d62728", 10, True)
    vec(170, 265, 270, 215, "#7b3fa0", 3, dashed=True)
    text(95, 375, 340, 52, "v_des: path tangent + cross-track correction at THIS trailer.\nThe angle to its heading sets the wheel differential.", "#7b3fa0", 10)
    # travel
    vec(780, 265, 880, 265, "#333333", 3); text(790, 240, 100, 20, "travel", font=10)

    # right-hand panel: signals and rules
    rect(910, 60, 330, 120, "LOAD CELLS (fixed end of each cable)\nsum = T_tl+T_tr+T_bl+T_br\nyaw moment = (T_tl+T_bl) - (T_tr+T_br)\npitch moment = (T_tl+T_tr) - (T_bl+T_br)", "#fff2d6", "#d6a000", 11, "left")
    rect(910, 200, 330, 150, "RULES\n1. Hitch = preloaded cables: a spring-like restoring torque, NOT an actuator.\n2. Wheels steer the trailer toward v_des.\n3. Tow, don't push: keep every hitch in tension.\n4. COMPRESSION IS A FAILURE: sum < baseline - 30 N => slow trailer j+1 at once (retension_guard); release over 1 s.", "#fde4e4", "#c0392b", 11, "left")
    rect(910, 370, 330, 110, "OPTIONAL bar sensor (axial force through the bar)\nCompare with the cable sum: python compare_loads.py\nIf the cable sum tracks it, the bar sensor is not needed.", "#ece4f7", "#7b3fa0", 11, "left")
    rect(60, 450, 700, 60, "Bar length is GEOMETRY (how far the trailers cut corners), not a force actuator. Keep it slow; keep forces in the cables and the wheels.", "#f5f5f5", "#999999", 11, "left")
    xml = ('<mxfile host="app.diagrams.net"><diagram name="Forces on one hitch" id="agv2"><mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" '
           'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1300" pageHeight="560" math="0" shadow="0">'
           '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cs) + "".join(es) + '</root></mxGraphModel></diagram></mxfile>')
    Path("docs/hitch_forces.drawio").write_text(xml, encoding="utf-8")
    print("wrote docs/hitch_forces.drawio:", len(cs), "shapes,", len(es), "arrows")


_forces_diagram()
