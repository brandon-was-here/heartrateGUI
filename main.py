#!/usr/bin/env python3
"""
Heart Monitor GUI (PySide6 + pyqtgraph)
- Waveform strip chart (PPG samples)
- Big BPM display
- Toolbar: port, baud, connect, record, snapshot
- Status/log footer
Placeholders are marked with TODO.
"""

from __future__ import annotations

import sys
import time
import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Deque, Tuple
from collections import deque

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Optional import guard for pyserial so app can run without it
try:
    import serial
    import serial.tools.list_ports as list_ports
except Exception:  # pragma: no cover
    serial = None
    list_ports = None


# ----------------------------
# App-level constants / config
# ----------------------------

APP_NAME = "Heart Monitor"
DATA_DIR = Path("./data")
PLOTS_DIR = DATA_DIR / "plots"
CONFIG_FILE = Path("./config.json")

DATA_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


# ----------------------------
# Simple config model
# ----------------------------

@dataclass
class AppConfig:
    last_port: str = ""
    baud: int = 115200
    smoothing_window: int = 5
    simulate: bool = True

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_FILE.exists():
            try:
                return cls(**json.loads(CONFIG_FILE.read_text()))
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))


# ----------------------------
# Serial worker (QThread)
# ----------------------------

class SerialWorker(QtCore.QThread):
    """Reads lines from serial OR generates simulated data."""
    line_received = QtCore.Signal(str)
    sample_received = QtCore.Signal(float)      # raw sample
    bpm_received = QtCore.Signal(float)         # bpm value
    status = QtCore.Signal(str)

    def __init__(self, port: str, baud: int, simulate: bool = False, parent=None):
        super().__init__(parent)
        self._port = port
        self._baud = baud
        self._simulate = simulate
        self._running = False
        self._ser: Optional["serial.Serial"] = None

        # Sim generator state
        self._t = 0.0
        self._fs = 120.0  # Hz
        self._dt = 1.0 / self._fs
        self._bpm_sim = 72.0

    def run(self):
        self._running = True
        if self._simulate:
            self.status.emit("Simulating data …")
            self._run_simulated()
        else:
            if serial is None:
                self.status.emit("pyserial not installed; cannot open serial.")
                return
            try:
                self._ser = serial.Serial(self._port, self._baud, timeout=1)
                self.status.emit(f"Opened {self._port} @ {self._baud}")
                self._run_serial()
            except Exception as e:
                self.status.emit(f"Serial open failed: {e}")

    def stop(self):
        self._running = False
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    def _run_serial(self):
        """Read line-oriented protocol. Expected examples:
           'S:512'  (sample)
           'BPM:78' (bpm)
        """
        while self._running and self._ser:
            try:
                raw = self._ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue
                self.line_received.emit(raw)

                if raw.startswith("S:"):
                    v = float(raw.split(":", 1)[1])
                    self.sample_received.emit(v)
                elif raw.upper().startswith("BPM:"):
                    v = float(raw.split(":", 1)[1])
                    self.bpm_received.emit(v)
            except Exception as e:
                self.status.emit(f"Serial read error: {e}")
                time.sleep(0.1)

    def _run_simulated(self):
        """Generate a quasi-PPG plus BPM updates."""
        last_bpm_emit = time.time()
        while self._running:
            # PPG-like waveform: base sine + pulses
            heart_hz = self._bpm_sim / 60.0
            v = 520 + 80*np.sin(2*np.pi*heart_hz*self._t) \
                + 30*np.sin(2*np.pi*2*heart_hz*self._t + 0.8) \
                + np.random.normal(0, 3.0)

            self.sample_received.emit(float(v))

            # Occasional BPM updates (1 Hz)
            now = time.time()
            if now - last_bpm_emit > 1.0:
                jitter = np.random.normal(0, 1.5)
                self.bpm_received.emit(float(max(40.0, self._bpm_sim + jitter)))
                last_bpm_emit = now

            self._t += self._dt
            self.msleep(int(self._dt * 1000))


# ----------------------------
# Plot widget
# ----------------------------

class StripChart(pg.PlotWidget):
    """Rolling-time plot for samples."""
    def __init__(self, window_seconds: float = 12.0, fs_hint: float = 120.0, parent=None):
        super().__init__(parent=parent, background="w")
        self.setTitle("PPG Waveform", color="#333", size="12pt")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel("left", "Amplitude (arb.)")
        self.setLabel("bottom", "Time (s)")

        self._window_s = window_seconds
        self._fs = fs_hint
        self._maxlen = max(10, int(self._window_s * self._fs))

        self._buf: Deque[float] = deque(maxlen=self._maxlen)
        self._t0 = time.time()

        pen = pg.mkPen(width=2)
        self._curve = self.plot([], [], pen=pen)

    def append(self, sample: float):
        self._buf.append(sample)

    def refresh(self):
        if not self._buf:
            return
        ys = np.frombuffer(np.array(self._buf, dtype=float), dtype=float)
        n = len(ys)
        xs = np.linspace(-self._window_s, 0.0, n)
        self._curve.setData(xs, ys)


# ----------------------------
# Main Window
# ----------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 680)

        self.cfg = cfg
        self.worker: Optional[SerialWorker] = None
        self.recording = False
        self.record_path: Optional[Path] = None
        self.csv_writer: Optional[csv.writer] = None
        self.csv_file = None
        self._last_samples_ts = 0.0

        # ----- Central layout -----
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(10)

        # Left: strip chart
        self.chart = StripChart(window_seconds=12.0)
        main_layout.addWidget(self.chart, stretch=3)

        # Right: info panel
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        main_layout.addLayout(right, stretch=1)

        # BPM box
        self.bpm_box = QtWidgets.QGroupBox("BPM")
        bpm_layout = QtWidgets.QHBoxLayout(self.bpm_box)
        self.lbl_bpm = QtWidgets.QLabel("---")
        self.lbl_bpm.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm.setStyleSheet("font-size: 42px; font-weight: 600;")
        bpm_layout.addWidget(self.lbl_bpm)
        right.addWidget(self.bpm_box)

        # Stats box (placeholders)
        self.stats_box = QtWidgets.QGroupBox("Stats")
        stats_layout = QtWidgets.QFormLayout(self.stats_box)
        self.lbl_bpm_min = QtWidgets.QLabel("--")
        self.lbl_bpm_max = QtWidgets.QLabel("--")
        self.lbl_bpm_avg = QtWidgets.QLabel("--")
        self.lbl_quality = QtWidgets.QLabel("—")
        stats_layout.addRow("Min", self.lbl_bpm_min)
        stats_layout.addRow("Max", self.lbl_bpm_max)
        stats_layout.addRow("Avg", self.lbl_bpm_avg)
        stats_layout.addRow("Signal", self.lbl_quality)
        right.addWidget(self.stats_box)

        # Notes
        self.notes_box = QtWidgets.QGroupBox("Session Notes")
        notes_layout = QtWidgets.QVBoxLayout(self.notes_box)
        self.edit_notes = QtWidgets.QPlainTextEdit()
        self.edit_notes.setPlaceholderText("Notes…")
        notes_layout.addWidget(self.edit_notes)
        right.addWidget(self.notes_box, stretch=1)

        # ----- Toolbar -----
        tb = QtWidgets.QToolBar("Toolbar")
        self.addToolBar(tb)

        self.cmb_port = QtWidgets.QComboBox()
        self.cmb_baud = QtWidgets.QComboBox()
        self.cmb_baud.addItems(["9600", "57600", "115200"])
        self.cmb_baud.setCurrentText(str(self.cfg.baud))
        self.chk_sim = QtWidgets.QCheckBox("Simulate")
        self.chk_sim.setChecked(self.cfg.simulate)

        self.btn_refresh = QtWidgets.QToolButton(text="Refresh")
        self.btn_connect = QtWidgets.QToolButton(text="Connect")
        self.btn_record = QtWidgets.QToolButton(text="Record")
        self.btn_snapshot = QtWidgets.QToolButton(text="Snapshot")

        tb.addWidget(QtWidgets.QLabel("Port: "))
        tb.addWidget(self.cmb_port)
        tb.addWidget(self.btn_refresh)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel("Baud: "))
        tb.addWidget(self.cmb_baud)
        tb.addSeparator()
        tb.addWidget(self.chk_sim)
        tb.addSeparator()
        tb.addWidget(self.btn_connect)
        tb.addWidget(self.btn_record)
        tb.addWidget(self.btn_snapshot)

        # ----- Status bar -----
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        # ----- Connections -----
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.toggle_connect)
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_snapshot.clicked.connect(self.save_snapshot)

        # Chart timer (UI refresh)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.chart.refresh)
        self._timer.start(33)  # ~30 FPS

        # Init UI state
        self.refresh_ports(initial=True)
        self.update_record_ui(False)

        # BPM running stats (placeholders)
        self._bpm_vals: Deque[float] = deque(maxlen=600)

    # ----- UI Helpers -----

    def refresh_ports(self, initial: bool = False):
        self.cmb_port.clear()
        ports = []
        if list_ports:
            try:
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []
        # Always allow manual text entry
        self.cmb_port.setEditable(True)
        self.cmb_port.addItems(ports)
        if initial and self.cfg.last_port:
            self.cmb_port.setCurrentText(self.cfg.last_port)

    def toggle_connect(self):
        if self.worker and self.worker.isRunning():
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        port = self.cmb_port.currentText().strip()
        baud = int(self.cmb_baud.currentText())
        simulate = self.chk_sim.isChecked()

        self.cfg.last_port = port
        self.cfg.baud = baud
        self.cfg.simulate = simulate
        self.cfg.save()

        self.worker = SerialWorker(port=port, baud=baud, simulate=simulate)
        self.worker.line_received.connect(self.on_line)
        self.worker.sample_received.connect(self.on_sample)
        self.worker.bpm_received.connect(self.on_bpm)
        self.worker.status.connect(self.status.showMessage)

        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.btn_connect.setText("Disconnect")
        self.status.showMessage(f"Connecting ({'sim' if simulate else port}) …")

    def disconnect_serial(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
        self.worker = None
        self.btn_connect.setText("Connect")
        self.status.showMessage("Disconnected")

    def on_worker_finished(self):
        self.btn_connect.setText("Connect")
        self.status.showMessage("Stopped")

    # ----- Data handlers -----

    def on_line(self, text: str):
        # Optional: show last line in status
        pass  # TODO: surface raw protocol in a log view if desired

    def on_sample(self, v: float):
        # Optional: basic smoothing (placeholder)
        # TODO: replace with your preferred filter
        self.chart.append(v)

        if self.recording:
            ts = time.time()
            # Write every N ms (throttle) or all samples; keep simple here:
            self.csv_writer.writerow([ts, "S", f"{v:.3f}"])

    def on_bpm(self, bpm: float):
        self.lbl_bpm.setText(f"{bpm:.0f}")
        self._bpm_vals.append(bpm)

        # Update placeholder stats
        arr = np.array(self._bpm_vals) if self._bpm_vals else np.array([])
        if arr.size:
            self.lbl_bpm_min.setText(f"{arr.min():.0f}")
            self.lbl_bpm_max.setText(f"{arr.max():.0f}")
            self.lbl_bpm_avg.setText(f"{arr.mean():.0f}")
            # TODO: compute real signal quality; placeholder heuristic:
            self.lbl_quality.setText("OK")

        if self.recording:
            ts = time.time()
            self.csv_writer.writerow([ts, "BPM", f"{bpm:.2f}"])

    # ----- Recording / snapshot -----

    def toggle_record(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        tstamp = time.strftime("%Y%m%d_%H%M%S")
        self.record_path = DATA_DIR / f"session_{tstamp}.csv"
        self.csv_file = self.record_path.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "type", "value"])
        # Save minimal session header as JSON sidecar
        header = {
            "notes": self.edit_notes.toPlainText().strip(),
            "port": self.cmb_port.currentText().strip(),
            "baud": int(self.cmb_baud.currentText()),
            "simulate": self.chk_sim.isChecked(),
            "app": APP_NAME,
        }
        (self.record_path.with_suffix(".json")).write_text(json.dumps(header, indent=2))
        self.update_record_ui(True)
        self.status.showMessage(f"Recording → {self.record_path.name}")

    def stop_recording(self):
        try:
            if self.csv_file:
                self.csv_file.flush()
                self.csv_file.close()
        finally:
            self.csv_file = None
            self.csv_writer = None
            self.update_record_ui(False)
            self.status.showMessage("Recording stopped")

    def update_record_ui(self, on: bool):
        self.recording = on
        self.btn_record.setText("Stop" if on else "Record")

    def save_snapshot(self):
        tstamp = time.strftime("%Y%m%d_%H%M%S")
        path = PLOTS_DIR / f"plot_{tstamp}.png"
        exporter = ImageExporter(self.chart.plotItem)
        exporter.parameters()['width'] = 1200
        exporter.export(str(path))
        self.status.showMessage(f"Saved snapshot → {path}")


    # ----- Cleanup -----

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(500)
        if self.recording:
            self.stop_recording()
        super().closeEvent(e)


# --- Snapshot export support ---
from pyqtgraph.exporters import ImageExporter



def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    cfg = AppConfig.load()
    win = MainWindow(cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
