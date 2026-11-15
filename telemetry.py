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
        print(f"[Telemetry] UDP target: {self.addr[0]}:{self.addr[1]} | CSV enabled: {self.enable_csv}")


    def update(
        self,
        t: float,
        pos_xy,
        lead_hdg: float,
        joint_yaws: list[float],
        joint_eulers: list[tuple[float, float, float]],
        forces: dict[str, float],
    ) -> None:
        # Handle simulator Reset (t jumps backward to 0.0) so rate-limiter doesn't stall
        if t < self.last_stream:
            self.last_stream = -1.0

        if t - self.last_stream < 0.02:
            return
        self.last_stream = t

        row = {
            "timestamp": round(float(t), 4),
            "pose/x_m": round(float(pos_xy[0]), 3),
            "pose/y_m": round(float(pos_xy[1]), 3),
            "pose/hdg_deg": round(float(np.rad2deg(lead_hdg)), 3),
            "tension/peak_N": round(float(max(forces.values())) if forces else 0.0, 2),
        }

        # Automatically emits joint1/..., joint2/..., ..., jointN/... for any number of trailers
        for idx, (y_cmd, (yaw_meas, pitch_meas, roll_meas)) in enumerate(
            zip(joint_yaws, joint_eulers), start=1
        ):
            row[f"joint{idx}/yaw_cmd_deg"] = round(float(np.rad2deg(y_cmd)), 3)
            row[f"joint{idx}/yaw_meas_deg"] = round(float(np.rad2deg(yaw_meas)), 3)
            row[f"joint{idx}/pitch_meas_deg"] = round(float(np.rad2deg(pitch_meas)), 3)
            row[f"joint{idx}/roll_meas_deg"] = round(float(np.rad2deg(roll_meas)), 3)

        for t_name, f_val in forces.items():
            row[f"tension/{t_name}_N"] = round(float(f_val), 2)

        self.sock.sendto(json.dumps(row).encode("utf-8"), self.addr)
        self.sock.sendto(json.dumps(row).encode("utf-8"), self.addr)
        if not hasattr(self, "_printed_first_pkt"):
            print(f"[Telemetry] First packet sent ({len(row)} fields)")
            self._printed_first_pkt = True
        
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
