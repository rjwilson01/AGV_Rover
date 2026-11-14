import asyncio
import csv
import json
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False

USE_IR_CAMERA_FEEDBACK = True
WS_HOST = "127.0.0.1"
WS_PORT = 9870

def calc_cable_len(yaw, pitch, roll, y_sign, z_sign, L_bar=0.35, W_y=0.24, H_z=0.165):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    bx, by, bz = L_bar, y_sign * W_y, z_sign * H_z
    wx = (cy * cp) * bx + (cy * sp * sr - sy * cr) * by + (cy * sp * cr + sy * sr) * bz
    wy = (sy * cp) * bx + (sy * sp * sr + cy * cr) * by + (sy * sp * cr - cy * sr) * bz
    wz = (-sp) * bx + (cp * sr) * by + (cp * cr) * bz
    return float(np.linalg.norm([wx + L_bar, wy - by, wz - bz]))

def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(yaw), float(pitch), float(roll)

class PlotJugglerWSBridge:
    def __init__(self, host="127.0.0.1", port=9870):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        if HAS_WS:
            t = threading.Thread(target=self._run_loop, daemon=True)
            t.start()

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _handler(self, websocket):
        self.clients.add(websocket)
        try:
            adv = json.dumps({
                "op": "advertise",
                "topic": "/agv_snake",
                "type": "json"
            })
            await websocket.send(adv)
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

    def publish(self, payload_dict):
        if not HAS_WS or not self.clients or self.loop is None:
            return
        msg = json.dumps({
            "op": "publish",
            "topic": "/agv_snake",
            "msg": payload_dict
        })
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send(msg), self.loop)

def main():
    base_dir = Path(__file__).resolve().parent
    xml_path = base_dir / "snake_agv_sim.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    bridge = PlotJugglerWSBridge(WS_HOST, WS_PORT)
    if HAS_WS:
        print(f"PlotJuggler Bridge WebSocket server listening on ws://{WS_HOST}:{WS_PORT}")
    else:
        print("Warning: 'websockets' not installed (pip install websockets). Writing sim_log.csv only.")

    capstan_names = [
        "cap1_tl", "cap1_tr", "cap1_bl", "cap1_br",
        "cap2_tl", "cap2_tr", "cap2_bl", "cap2_br",
    ]
    tendon_names = [
        "t1_tl", "t1_tr", "t1_bl", "t1_br",
        "t2_tl", "t2_tr", "t2_bl", "t2_br",
    ]
    wheel_names = ["m_t1_l", "m_t1_r", "m_t2_l", "m_t2_r", "m_t3_l", "m_t3_r"]

    for cname in capstan_names:
        data.actuator(cname).ctrl = 0.698

    signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    path_history = []
    L_SEG = 0.80
    TRACK_W = 0.64
    WHEEL_R = 0.18

    csv_rows = []
    dist_traveled = 0.0
    last_stream = 0.0
    last_print = time.time()
    interval_peak = 0.0
    turn_complete = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = float(data.time)

            _, lead_pitch, lead_hdg = quat_to_euler(data.joint("root_jnt").qpos[3:7])
            lead_hdg_deg = np.rad2deg(lead_hdg)

            v_center = 1.15
            if t < 5.0 or turn_complete:
                target_yaw = 0.0
            else:
                ramp = min(1.0, (t - 5.0) / 1.5)
                target_yaw = np.deg2rad(44.0) * ramp
                if abs(lead_hdg_deg) >= 176.0:
                    turn_complete = True
                    target_yaw = 0.0

            dist_traveled += v_center * model.opt.timestep
            path_history.append((dist_traveled, target_yaw))

            def lookup_yaw(delay_m):
                target_d = dist_traveled - delay_m
                for d, y in reversed(path_history):
                    if d <= target_d:
                        return y
                return 0.0

            y1 = lookup_yaw(0.40)
            y2 = lookup_yaw(2.00)

            seg_yaws = [y1, 0.5 * (y1 + y2), y2]
            for s_idx, psi in enumerate(seg_yaws):
                if abs(psi) < 1e-4:
                    wl = wr = v_center / WHEEL_R
                else:
                    R = L_SEG / np.tan(psi / 2.0)
                    wl = (v_center * (1.0 - TRACK_W / (2.0 * R))) / WHEEL_R
                    wr = (v_center * (1.0 + TRACK_W / (2.0 * R))) / WHEEL_R
                data.actuator(wheel_names[2 * s_idx]).ctrl = wl
                data.actuator(wheel_names[2 * s_idx + 1]).ctrl = wr

            _, p1_m, r1_m = quat_to_euler(data.joint("ball_joint_1").qpos)
            _, p2_m, r2_m = quat_to_euler(data.joint("ball_joint_2").qpos)
            j1_yaw_m, _, _ = quat_to_euler(data.joint("ball_joint_1").qpos)
            j2_yaw_m, _, _ = quat_to_euler(data.joint("ball_joint_2").qpos)

            if not USE_IR_CAMERA_FEEDBACK:
                p1_ctrl = r1_ctrl = p2_ctrl = r2_ctrl = 0.0
            else:
                p1_ctrl, r1_ctrl = p1_m, r1_m
                p2_ctrl, r2_ctrl = p2_m, r2_m

            for k, (ys, zs) in enumerate(signs):
                data.actuator(capstan_names[k]).ctrl = calc_cable_len(-y1, p1_ctrl, r1_ctrl, ys, zs) - 0.002
                data.actuator(capstan_names[4 + k]).ctrl = calc_cable_len(-y2, p2_ctrl, r2_ctrl, ys, zs) - 0.002

            mujoco.mj_step(model, data)
            viewer.sync()

            forces = {
                t_name: abs(float(data.actuator(c_name).force[0]))
                for c_name, t_name in zip(capstan_names, tendon_names)
            }
            live_peak = max(forces.values())
            if live_peak > interval_peak:
                interval_peak = live_peak

            if t - last_stream >= 0.02:
                last_stream = t
                row = {
                    "timestamp": round(t, 4),
                    "pose/hdg_deg": round(lead_hdg_deg, 3),
                    "joint1/yaw_cmd_deg": round(np.rad2deg(y1), 3),
                    "joint1/yaw_meas_deg": round(np.rad2deg(j1_yaw_m), 3),
                    "joint1/pitch_meas_deg": round(np.rad2deg(p1_m), 3),
                    "joint1/roll_meas_deg": round(np.rad2deg(r1_m), 3),
                    "joint2/yaw_cmd_deg": round(np.rad2deg(y2), 3),
                    "joint2/yaw_meas_deg": round(np.rad2deg(j2_yaw_m), 3),
                    "tension/peak_N": round(live_peak, 2),
                }
                for t_name, f_val in forces.items():
                    row[f"tension/{t_name}_N"] = round(f_val, 2)
                bridge.publish(row)
                csv_rows.append(row)

            if time.time() - last_print > 0.6:
                max_cable = max(forces, key=forces.get)
                print(
                    f"t={t:4.1f}s | Hdg={lead_hdg_deg:+6.1f} deg | "
                    f"J1={np.rad2deg(y1):+5.1f}/{np.rad2deg(j1_yaw_m):+5.1f} deg | "
                    f"Peak={interval_peak:6.1f} N [{max_cable}]"
                )
                interval_peak = 0.0
                last_print = time.time()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    if csv_rows:
        csv_path = base_dir / "sim_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved {len(csv_rows)} samples to {csv_path}")

if __name__ == "__main__":
    main()
