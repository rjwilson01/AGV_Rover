import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
import xml.etree.ElementTree as ET
import ctypes


def _find_window_by_pid(pid: int) -> tuple[int, tuple[int, int, int, int]] | None:
    """Returns (hwnd, (left, top, right, bottom)) for the main visible window of `pid`."""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        win_pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(win_pid))
        if win_pid.value == pid:
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            # Ignore tiny hidden/helper windows
            if (rect.right - rect.left) > 400 and (rect.bottom - rect.top) > 300:
                found.append((hwnd, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    user32.EnumWindows(enum_cb, 0)
    return found[0] if found else None


import ctypes
import ctypes.wintypes
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
import xml.etree.ElementTree as ET


def build_plotjuggler_layout(layout_path: Path, udp_port: int = 9870, num_joints: int = 1, bar_yaw: bool = False) -> Path:
    root = ET.Element(
        "root",
        binding="source",
        format="PlotJuggler",
        pj4_version="4",
    )

    tabbed = ET.SubElement(root, "tabbed_widget", id="", name="main", parent="main_window")

    def add_tab(tab_name: str, panels: list[tuple[str, list[tuple[str, str]]]]):
        tab = ET.SubElement(
            tabbed,
            "Tab",
            containers="1",
            id=str(uuid.uuid4()),
            tab_name=tab_name,
        )
        container = ET.SubElement(tab, "Container")
        sizes = ";".join([f"{1.0 / len(panels):.8f}"] * len(panels))
        splitter = ET.SubElement(
            container,
            "DockSplitter",
            count=str(len(panels)),
            orientation="-",
            sizes=sizes,
        )

        for title, curves in panels:
            dock = ET.SubElement(
                splitter,
                "DockArea",
                id=str(uuid.uuid4()),
                name=title,
            )
            plot = ET.SubElement(
                dock,
                "plot",
                id=str(uuid.uuid4()),
                line_width="1.0",
                mode="TimeSeries",
                style="Lines",
                title="",
                tracker_enabled="true",
            )
            ET.SubElement(
                plot,
                "range",
                bottom="-1.0",
                left="0.0",
                left_ns="0",
                right="10.0",
                right_ns="10000000000",
                top="1.0",
                x_basis="absolute",
            )
            for field_name, color in curves:
                ET.SubElement(
                    plot,
                    "curve",
                    color=color,
                    dataset_source="[stream] UDP Server",
                    field=field_name.lstrip("/"),
                    markers_visible="false",
                    topic="udp/data",
                    visible="true",
                )

    # Tab 1: Vehicle Pose & Peak Tension
    add_tab("Overview", [
        ("Lead Heading (deg)", [("pose/hdg_deg", "#1f77b4")]),
        ("Peak Tension (N)", [("tension/peak_N", "#d62728")]),
    ])

    # Dynamic Tabs: One tab per joint (joint1..jointN)
    for j in range(1, num_joints + 1):
        add_tab(f"Joint {j}", [
            ("Yaw Tracking (deg)", [
                (f"joint{j}/yaw_cmd_deg", "#1f77b4"),
                (f"joint{j}/yaw_meas_deg", "#ff7f0e"),
            ]),
            # Yaw-tube builds only: commanded vs measured yaw of each drawbar tube, and the torque it takes
            *([
                ("Bar Yaw (deg)", [
                    (f"joint{j}/rear_yaw_cmd_deg", "#1f77b4"),
                    (f"joint{j}/rear_yaw_meas_deg", "#17becf"),
                    (f"joint{j}/front_yaw_cmd_deg", "#d62728"),
                    (f"joint{j}/front_yaw_meas_deg", "#ff7f0e"),
                    (f"joint{j}/ball_yaw_meas_deg", "#7f7f7f"),
                ]),
                ("Bar Yaw Torque (Nm)", [
                    (f"joint{j}/rear_yaw_torque_Nm", "#1f77b4"),
                    (f"joint{j}/front_yaw_torque_Nm", "#d62728"),
                ]),
            ] if bar_yaw else []),
            ("Pitch & Roll (deg)", [
                (f"joint{j}/pitch_meas_deg", "#2ca02c"),
                (f"joint{j}/roll_meas_deg", "#9467bd"),
            ]),
            ("Axle Distance Error (mm)", [
                (f"joint{j}/axle_dist_err_mm", "#d62728"),
            ]),
            ("Cable Tensions (N)", [
                (f"joint{j}/tension_tl_N", "#1f77b4"),
                (f"joint{j}/tension_tr_N", "#ff7f0e"),
                (f"joint{j}/tension_bl_N", "#2ca02c"),
                (f"joint{j}/tension_br_N", "#d62728"),
            ]),
        ])

    ET.SubElement(tabbed, "currentTabIndex", index="0")

    ET.SubElement(root, "data_processors")
    ET.SubElement(root, "right_panel_state", splitter_sizes="1569,0", style="2", width="1")
    ET.SubElement(
        root,
        "left_panel_state",
        sources_tab="stream",
        streaming_buffer="101",
        streaming_source="UDP Server",
    )
    ET.SubElement(
        root,
        "curve_list_state",
        custom_filter="",
        datasets_filter="",
        show_topics="true",
        show_values="false",
    )
    ET.SubElement(root, "chrome_state", main_splitter_sizes="400,1135")
    ET.SubElement(root, "pinned_toolboxes")

    layout_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(layout_path, encoding="UTF-8", xml_declaration=True)
    return layout_path


def _find_window_by_pid(pid: int) -> tuple[int, tuple[int, int, int, int]] | None:
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum_cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        win_pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(win_pid))
        if win_pid.value == pid:
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if (rect.right - rect.left) > 600 and (rect.bottom - rect.top) > 400:
                found.append((hwnd, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    user32.EnumWindows(enum_cb, 0)
    return found[0] if found else None

def _paste_text(text: str) -> None:
    subprocess.run("clip", input=text.encode("utf-16le"), check=True)

def _automate_pj4_startup(pid: int, layout_file: Path, timeout_s: float = 15.0) -> None:
    import pyautogui

    deadline = time.time() + timeout_s
    win_info = None
    while time.time() < deadline:
        win_info = _find_window_by_pid(pid)
        if win_info:
            break
        time.sleep(0.3)

    if not win_info:
        print("[PlotJuggler] Timed out waiting for PlotJuggler window.")
        return

    hwnd, (left, top, right, bottom) = win_info
    time.sleep(1.0)

    ctypes.windll.user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)

    # 1. Click Stream icon (Window Spy: 328, 79)
    pyautogui.click(left + 328, top + 79)
    time.sleep(0.25)

    # 2. Click '+' button next to UDP Server (Window Spy: 391, 114)
    pyautogui.click(left + 391, top + 114)
    time.sleep(0.5)

    # 3. Click 'OK' button on UDP Server popup (Window Spy Screen: 905, 597)
    pyautogui.click(905, 597)
    time.sleep(0.8)

    # 4. Load layout via Ctrl+L + clipboard paste
    # 4. Open Load Layout via File menu (Alt+F -> Down -> Enter)
    pyautogui.hotkey("alt", "f")
    time.sleep(0.3)
    pyautogui.press("down")
    time.sleep(0.15)
    pyautogui.press("enter")
    time.sleep(0.7)
  # Type full path directly and press Enter
    pyautogui.write(str(layout_file.resolve()), interval=0.01)
    time.sleep(0.1)
    pyautogui.press("enter")


def launch_plotjuggler(base_dir: Path, udp_port: int = 9870, num_joints: int = 1, bar_yaw: bool = False) -> subprocess.Popen:
    layout_file = build_plotjuggler_layout(base_dir / "pj_auto_layout.xml", udp_port, num_joints, bar_yaw)

    env = os.environ.copy()
    env["PLOTJUGGLER_ACCEPT_PREVIOUSLY_USED_STREAMING_PLUGIN"] = "1"
    env["PLOTJUGGLER_CHOOSE_EMPTY_PLACEHOLDERS_ON_MISSING_TIMESERIES"] = "1"

    # PlotJuggler is very chatty (graphics/plugin messages): keep it out of the sim terminal
    pj_log = open(base_dir / "plotjuggler.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [
            "C:/Program Files/PlotJuggler4/bin/PlotJuggler4.exe",
            "--nosplash",
        ],
        env=env,
        stdout=pj_log,
        stderr=subprocess.STDOUT,
    )

    # Run UI automation in a daemon thread so it doesn't block your simulation loop
    threading.Thread(
        target=_automate_pj4_startup,
        args=(proc.pid, layout_file),
        daemon=True,
    ).start()

    return proc
