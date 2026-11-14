import csv
import json
import socket
from pathlib import Path
import numpy as np


class TelemetryManager:
    def __init__(self, base_dir: Path, udp_host="127.0.0.1", udp_port=9870, enable_csv=False):
        self.addr = (udp_host, udp_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enable_csv = enable_csv
        self.csv_path = base_dir / "sim_log.csv"
        self.csv_rows = []
        self.last_stream = 0.0

    def update(self, t, pose_xy, lead_hdg_deg, y1, y2, j1_euler, j2_euler, forces):
        if t - self.last_stream < 0.02:
            return
        self.last_stream = t

        j1_yaw, j1_pitch, j1_roll = j1_euler
        j2_yaw, _, _ = j2_euler

        row = {
            "timestamp": round(t, 4),
            "pose/x_m": round(float(pose_xy[0]), 3),
            "pose/y_m": round(float(pose_xy[1]), 3),
            "pose/hdg_deg": round(lead_hdg_deg, 3),
            "joint1/yaw_cmd_deg": round(np.rad2deg(y1), 3),
            "joint1/yaw_meas_deg": round(np.rad2deg(j1_yaw), 3),
            "joint1/pitch_meas_deg": round(np.rad2deg(j1_pitch), 3),
            "joint1/roll_meas_deg": round(np.rad2deg(j1_roll), 3),
            "joint2/yaw_cmd_deg": round(np.rad2deg(y2), 3),
            "joint2/yaw_meas_deg": round(np.rad2deg(j2_yaw), 3),
            "tension/peak_N": round(max(forces.values()), 2),
        }
        for t_name, f_val in forces.items():
            row[f"tension/{t_name}_N"] = round(f_val, 2)

        self.sock.sendto(json.dumps(row).encode("utf-8"), self.addr)
        if self.enable_csv:
            self.csv_rows.append(row)

    def close(self):
        self.sock.close()
        if self.enable_csv and self.csv_rows:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.csv_rows)
            print(f"Saved {len(self.csv_rows)} samples to {self.csv_path}")
