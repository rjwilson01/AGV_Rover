import csv
import json
import socket
import subprocess
import time
from pathlib import Path
import numpy as np
from imu import tilt_from_accel
from pj_launcher import launch_plotjuggler



def _git_version(base_dir: Path) -> str:
    """Short commit hash, with '-dirty' if there are uncommitted changes."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=base_dir, capture_output=True, text=True, timeout=5).stdout.strip()
        h = run("rev-parse", "--short", "HEAD")
        return (h + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")) if h else "unknown"
    except Exception:
        return "unknown"


class TelemetryManager:
    def __init__(self, base_dir: Path, udp_host="127.0.0.1", udp_port=9870, enable_csv=False, launch_pj=True,
                 bar_yaw=False, meta=None):
        self.addr = (udp_host, udp_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.enable_csv = enable_csv
        self.meta = {"log_version": 1, "git_commit": _git_version(base_dir),
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})}
        self.csv_path = base_dir / "sim_log.csv"   # same file every run: just overwrite it, then git push
        self.csv_rows = []
        self.last_stream = 0.0
        self.segment_idx = 0
        self.segment_name = ""
        print(f"[Telemetry] UDP target: {self.addr[0]}:{self.addr[1]} | CSV enabled: {self.enable_csv}")
        self.pj_proc = (
            launch_plotjuggler(base_dir, udp_port=udp_port, num_joints=2, bar_yaw=bar_yaw) if launch_pj else None
        )

    def set_segment(self, idx: int, name: str) -> None:
        """Tag following rows with the test segment they belong to (1-based; 0 = not a test run)."""
        self.segment_idx, self.segment_name = idx, name
        self.last_stream = -1.0

    def update(
        self,
        t: float,
        pos_xy,
        lead_hdg: float,
        joint_yaws: list[float],
        joint_eulers: list[tuple[float, float, float]],
        forces: dict[str, float],
        extra_metrics: dict[str, float] | None = None,
        joint_cables: dict[str, tuple[float, float]] | None = None,
        trailer_positions: list[tuple[float, float, float]] | None = None,
        ir_readings: list[tuple[float, float, float]] | None = None,
        imu_readings=None,
        gnss=None,
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
            "test/segment": self.segment_idx,
        }
        if extra_metrics:
            row.update(extra_metrics)

        if gnss is not None:
            row["gnss/x_m"] = round(gnss.x, 3)
            row["gnss/y_m"] = round(gnss.y, 3)
            row["gnss/speed_mps"] = round(gnss.speed, 3)
            row["gnss/valid"] = int(gnss.valid)
            row["gnss/pos_err_m"] = round(float(np.hypot(gnss.x - pos_xy[0], gnss.y - pos_xy[1])), 3)
        if imu_readings is not None:
            for n, r in enumerate(imu_readings, start=1):
                for ax, v in zip("xyz", r.acc):
                    row[f"imu{n}/acc_{ax}_mps2"] = round(float(v), 4)
                for ax, v in zip("xyz", r.gyro):
                    row[f"imu{n}/gyro_{ax}_dps"] = round(float(np.rad2deg(v)), 4)
                p, ro = tilt_from_accel(r.acc)
                row[f"imu{n}/acc_pitch_deg"] = round(float(np.rad2deg(p)), 3)
                row[f"imu{n}/acc_roll_deg"] = round(float(np.rad2deg(ro)), 3)

        # Automatically emits joint1/..., joint2/..., ..., jointN/... for any number of trailers
        for idx, (y_cmd, (yaw_meas, pitch_meas, roll_meas)) in enumerate(
            zip(joint_yaws, joint_eulers), start=1
        ):
            row[f"joint{idx}/yaw_cmd_deg"] = round(float(np.rad2deg(y_cmd)), 3)
            row[f"joint{idx}/yaw_meas_deg"] = round(float(np.rad2deg(yaw_meas)), 3)
            row[f"joint{idx}/pitch_meas_deg"] = round(float(np.rad2deg(pitch_meas)), 3)
            row[f"joint{idx}/roll_meas_deg"] = round(float(np.rad2deg(roll_meas)), 3)

            # IR camera reading (what the controller sees) and its error vs the true joint angle
            if ir_readings is not None and idx - 1 < len(ir_readings):
                ir_yaw, ir_pitch, ir_roll = ir_readings[idx - 1]
                row[f"joint{idx}/ir_yaw_deg"] = round(float(np.rad2deg(ir_yaw)), 3)
                row[f"joint{idx}/ir_pitch_deg"] = round(float(np.rad2deg(ir_pitch)), 3)
                row[f"joint{idx}/ir_roll_deg"] = round(float(np.rad2deg(ir_roll)), 3)
                row[f"joint{idx}/ir_yaw_err_deg"] = round(float(np.rad2deg(ir_yaw - yaw_meas)), 3)
                row[f"joint{idx}/ir_pitch_err_deg"] = round(float(np.rad2deg(ir_pitch - pitch_meas)), 3)
                row[f"joint{idx}/ir_roll_err_deg"] = round(float(np.rad2deg(ir_roll - roll_meas)), 3)

            # 1. Derived axle-to-axle distance
            if trailer_positions is not None and idx < len(trailer_positions):
                r_meas = row.get(f"joint{idx}/rear_bar_meas_m")
                f_meas = row.get(f"joint{idx}/front_bar_meas_m")
                if r_meas is not None and f_meas is not None:
                    # Yaw-tube builds supply a geometry-based value (the formula below assumes straight bars)
                    d_kin = row.get(f"joint{idx}/axle_dist_kin_m")
                    if d_kin is None:
                        d_kin = float(np.sqrt(r_meas**2 + f_meas**2 + 2.0 * r_meas * f_meas * np.cos(yaw_meas)))
                    p_front = np.asarray(trailer_positions[idx - 1])
                    p_rear = np.asarray(trailer_positions[idx])
                    d_act = float(np.linalg.norm(p_front - p_rear))
                    row[f"joint{idx}/axle_dist_kin_m"] = round(d_kin, 4)
                    row[f"joint{idx}/axle_dist_act_m"] = round(d_act, 4)
                    row[f"joint{idx}/axle_dist_err_mm"] = round((d_act - d_kin) * 1000.0, 2)

            # 2. Per-joint cables (tl, tr, bl, br) grouped right under joint{idx}/...
            if joint_cables is not None and idx - 1 < len(joint_cables):
                for corner, (len_cmd, len_meas, f_val) in joint_cables[idx - 1].items():
                    row[f"joint{idx}/cable_{corner}_cmd_m"] = round(float(len_cmd), 4)
                    row[f"joint{idx}/cable_{corner}_meas_m"] = round(float(len_meas), 4)
                    row[f"joint{idx}/cable_{corner}_err_mm"] = round((float(len_meas) - float(len_cmd)) * 1000.0, 2)
                    row[f"joint{idx}/tension_{corner}_N"] = round(float(f_val), 2)


        for t_name, f_val in forces.items():
            row[f"tension/{t_name}_N"] = round(float(f_val), 2)

        self.sock.sendto(json.dumps(row).encode("utf-8"), self.addr)
        if not hasattr(self, "_printed_first_pkt"):
            print(f"[Telemetry] First packet sent ({len(row)} fields)")
            self._printed_first_pkt = True
        
        if self.enable_csv:
            self.csv_rows.append({**row, "test/name": self.segment_name} if self.segment_idx else row)

    def flush_csv(self) -> None:
        """(Re)write the CSV with everything logged so far, so a crash mid-sequence loses nothing."""
        if not (self.enable_csv and self.csv_rows):
            return
        fields = list(dict.fromkeys(k for r in self.csv_rows for k in r))
        with open(self.csv_path, "w", newline="") as f:
            for k, v in self.meta.items():
                f.write(f"# {k}: {v}\n")   # read with pandas.read_csv(path, comment="#")
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(self.csv_rows)

    def close(self):
        self.sock.close()
        if self.pj_proc is not None:
            print(f"[Telemetry] Terminate PJ")
            self.pj_proc.terminate()
        if self.enable_csv and self.csv_rows:
            self.flush_csv()
            print(f"Saved {len(self.csv_rows)} samples to {self.csv_path}")
