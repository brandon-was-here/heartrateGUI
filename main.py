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

# Import the real worker
from worker import SerialWorker

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

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                return cls(**data)
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))


# ----------------------------
# Note: SerialWorker is imported from worker.py
# ----------------------------


# ----------------------------
# Plot widget
# ----------------------------

class StripChart(pg.PlotWidget):
    """Rolling-time plot for samples."""
    def __init__(self, window_seconds: float = 12.0, fs_hint: float = 120.0, parent=None):
        super().__init__(parent=parent, background="white")
        self.setTitle("PPG Waveform", color="#333", size="14pt")
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel("left", "Amplitude (arb.)", **{"font-size": "11pt", "color": "#333"})
        self.setLabel("bottom", "Time (s)", **{"font-size": "11pt", "color": "#333"})
        
        # Style the plot with better contrast
        self.getAxis("left").setPen(pg.mkPen(color="#666", width=1))
        self.getAxis("bottom").setPen(pg.mkPen(color="#666", width=1))
        self.getAxis("left").setTextPen(pg.mkPen(color="#333"))
        self.getAxis("bottom").setTextPen(pg.mkPen(color="#333"))

        self._window_s = window_seconds
        self._fs = fs_hint
        self._maxlen = max(10, int(self._window_s * self._fs))

        self._buf: Deque[float] = deque(maxlen=self._maxlen)
        self._t0 = time.time()

        # Vibrant waveform color with strong contrast
        pen = pg.mkPen(color="#00A86B", width=3)  # Emerald green
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
        self.resize(1400, 800)
        
        # Set window background color
        self.setStyleSheet("QMainWindow { background-color: #F5F5F5; }")

        self.cfg = cfg
        self.worker: Optional[SerialWorker] = None
        self.recording = False
        self.record_path: Optional[Path] = None
        self.csv_writer: Optional[csv.writer] = None
        self.csv_file = None
        self._last_samples_ts = 0.0
        self._connected = False  # Track connection state
        
        # BPM calculation state
        self._bpm_vals: Deque[float] = deque(maxlen=600)
        self._peak_times: Deque[float] = deque(maxlen=10)  # timestamps of last 10 peaks
        self._last_value = 0
        self._last_was_rising = False
        self._samples_for_bpm: Deque[Tuple[float, float]] = deque(maxlen=500)  # (timestamp, value)
        self._settling_samples = 0  # Count samples to ignore during sensor settling
        self._has_good_signal = False       # True only when finger is on and waveform looks real
        self._no_signal_samples = 0         # how long we've been in "no-signal / low" territory

        # ----- Toolbar (Header with grouped controls) -----
        self._create_toolbar()

        # ----- Central layout -----
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Chart container with recording indicator overlay
        chart_container = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)
        
        # Create chart with enhanced styling
        self.chart = StripChart(window_seconds=12.0)
        
        # Recording indicator overlay (positioned in top-right of chart)
        self.recording_indicator = QtWidgets.QLabel("⏺ RECORDING")
        self.recording_indicator.setStyleSheet("""
            QLabel {
                background-color: #FF3333;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
        """)
        self.recording_indicator.setVisible(False)
        self.recording_indicator.setParent(self.chart)
        self.recording_indicator.move(10, 10)
        
        chart_layout.addWidget(self.chart)
        main_layout.addWidget(chart_container, stretch=4)

        # Bottom panel: BPM statistics + Session notes
        bottom_panel = QtWidgets.QHBoxLayout()
        bottom_panel.setSpacing(12)
        
        # BPM Statistics Panel (horizontal layout)
        stats_container = QtWidgets.QGroupBox("BPM Statistics")
        stats_container.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        stats_layout = QtWidgets.QHBoxLayout(stats_container)
        stats_layout.setSpacing(20)
        stats_layout.setContentsMargins(15, 15, 15, 15)
        
        # Current BPM (large display)
        current_bpm_layout = QtWidgets.QVBoxLayout()
        current_bpm_layout.setSpacing(2)
        lbl_current_title = QtWidgets.QLabel("Current")
        lbl_current_title.setStyleSheet("font-size: 10pt; color: #666; font-weight: normal;")
        lbl_current_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm = QtWidgets.QLabel("--")
        self.lbl_bpm.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm.setStyleSheet("font-size: 48px; font-weight: 700; color: #0066CC;")
        current_bpm_layout.addWidget(lbl_current_title)
        current_bpm_layout.addWidget(self.lbl_bpm)
        stats_layout.addLayout(current_bpm_layout)
        
        # Vertical separator
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.VLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        stats_layout.addWidget(sep1)
        
        # Min BPM
        min_layout = QtWidgets.QVBoxLayout()
        min_layout.setSpacing(2)
        lbl_min_title = QtWidgets.QLabel("Minimum")
        lbl_min_title.setStyleSheet("font-size: 10pt; color: #666; font-weight: normal;")
        lbl_min_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm_min = QtWidgets.QLabel("--")
        self.lbl_bpm_min.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm_min.setStyleSheet("font-size: 24px; font-weight: 600; color: #333;")
        min_layout.addWidget(lbl_min_title)
        min_layout.addWidget(self.lbl_bpm_min)
        stats_layout.addLayout(min_layout)
        
        # Max BPM
        max_layout = QtWidgets.QVBoxLayout()
        max_layout.setSpacing(2)
        lbl_max_title = QtWidgets.QLabel("Maximum")
        lbl_max_title.setStyleSheet("font-size: 10pt; color: #666; font-weight: normal;")
        lbl_max_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm_max = QtWidgets.QLabel("--")
        self.lbl_bpm_max.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm_max.setStyleSheet("font-size: 24px; font-weight: 600; color: #333;")
        max_layout.addWidget(lbl_max_title)
        max_layout.addWidget(self.lbl_bpm_max)
        stats_layout.addLayout(max_layout)
        
        # Average BPM
        avg_layout = QtWidgets.QVBoxLayout()
        avg_layout.setSpacing(2)
        lbl_avg_title = QtWidgets.QLabel("Average")
        lbl_avg_title.setStyleSheet("font-size: 10pt; color: #666; font-weight: normal;")
        lbl_avg_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm_avg = QtWidgets.QLabel("--")
        self.lbl_bpm_avg.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_bpm_avg.setStyleSheet("font-size: 24px; font-weight: 600; color: #333;")
        avg_layout.addWidget(lbl_avg_title)
        avg_layout.addWidget(self.lbl_bpm_avg)
        stats_layout.addLayout(avg_layout)
        
        # Vertical separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.VLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        stats_layout.addWidget(sep2)
        
        # Signal Quality
        quality_layout = QtWidgets.QVBoxLayout()
        quality_layout.setSpacing(2)
        lbl_quality_title = QtWidgets.QLabel("Signal")
        lbl_quality_title.setStyleSheet("font-size: 10pt; color: #666; font-weight: normal;")
        lbl_quality_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_quality = QtWidgets.QLabel("--")
        self.lbl_quality.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_quality.setStyleSheet("font-size: 18px; font-weight: 600; color: #00A86B;")
        quality_layout.addWidget(lbl_quality_title)
        quality_layout.addWidget(self.lbl_quality)
        stats_layout.addLayout(quality_layout)
        
        bottom_panel.addWidget(stats_container, stretch=3)
        
        # Session Notes (always visible, no scrolling)
        notes_container = QtWidgets.QGroupBox("Session Notes")
        notes_container.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        notes_layout = QtWidgets.QVBoxLayout(notes_container)
        notes_layout.setContentsMargins(10, 10, 10, 10)
        self.edit_notes = QtWidgets.QPlainTextEdit()
        self.edit_notes.setPlaceholderText("Enter session notes here...")
        self.edit_notes.setStyleSheet("""
            QPlainTextEdit {
                border: 1px solid #CCC;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                font-size: 11pt;
            }
        """)
        # Fixed height to ensure always visible
        self.edit_notes.setMaximumHeight(150)
        self.edit_notes.setMinimumHeight(150)
        notes_layout.addWidget(self.edit_notes)
        bottom_panel.addWidget(notes_container, stretch=2)
        
        main_layout.addLayout(bottom_panel, stretch=1)

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
        
        # BPM calculation timer
        self._bpm_timer = QtCore.QTimer(self)
        self._bpm_timer.timeout.connect(self.calculate_bpm)
        self._bpm_timer.start(1000)  # Calculate BPM every second

        # Init UI state
        self.refresh_ports(initial=True)
        self.update_record_ui(False)
    
    def _create_toolbar(self):
        """Create toolbar with grouped port selection and recording controls."""
        tb = QtWidgets.QToolBar("Controls")
        tb.setMovable(False)
        tb.setStyleSheet("""
            QToolBar {
                background-color: #E8E8E8;
                border-bottom: 2px solid #CCC;
                spacing: 3px;
                padding: 4px;
            }
            QLabel {
                font-weight: bold;
                font-size: 10pt;
                padding-right: 2px;
            }
            QComboBox {
                min-width: 120px;
                padding: 4px;
            }
            QToolButton {
                background-color: #4A90E2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10pt;
                min-width: 80px;
            }
            QToolButton:hover {
                background-color: #357ABD;
            }
            QToolButton:pressed {
                background-color: #2E6DA4;
            }
        """)
        self.addToolBar(tb)

        # Port Selection Group
        lbl_port = QtWidgets.QLabel("Port:")
        self.cmb_port = QtWidgets.QComboBox()
        self.cmb_port.setMinimumWidth(150)
        self.cmb_port.setToolTip("Select serial port for heart rate monitor")
        
        self.btn_refresh = QtWidgets.QToolButton()
        self.btn_refresh.setText("🔄 Refresh")
        self.btn_refresh.setToolTip("Refresh available serial ports")
        
        tb.addWidget(lbl_port)
        tb.addWidget(self.cmb_port)
        tb.addWidget(self.btn_refresh)
        
        # Separator
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.VLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep1.setStyleSheet("margin: 0 8px;")
        tb.addWidget(sep1)
        
        # Baud Rate Group
        lbl_baud = QtWidgets.QLabel("Baud:")
        self.cmb_baud = QtWidgets.QComboBox()
        self.cmb_baud.addItems(["9600", "57600", "115200"])
        self.cmb_baud.setCurrentText(str(self.cfg.baud))
        self.cmb_baud.setToolTip("Select baud rate for serial communication")
        
        tb.addWidget(lbl_baud)
        tb.addWidget(self.cmb_baud)
        
        # Separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.VLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep2.setStyleSheet("margin: 0 8px;")
        tb.addWidget(sep2)
        
        # Recording Controls Group
        self.btn_connect = QtWidgets.QToolButton()
        self.btn_connect.setText("▶ Connect")
        self.btn_connect.setToolTip("Connect to heart rate monitor")
        
        self.btn_record = QtWidgets.QToolButton()
        self.btn_record.setText("⏺ Record")
        self.btn_record.setToolTip("Start/stop recording session data")
        self.btn_record.setStyleSheet("""
            QToolButton {
                background-color: #E74C3C;
            }
            QToolButton:hover {
                background-color: #C0392B;
            }
        """)
        
        self.btn_snapshot = QtWidgets.QToolButton()
        self.btn_snapshot.setText("📷 Snapshot")
        self.btn_snapshot.setToolTip("Save current waveform as image")
        self.btn_snapshot.setStyleSheet("""
            QToolButton {
                background-color: #95A5A6;
            }
            QToolButton:hover {
                background-color: #7F8C8D;
            }
        """)
        
        tb.addWidget(self.btn_connect)
        tb.addWidget(self.btn_record)
        tb.addWidget(self.btn_snapshot)

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
        if self._connected and self.worker:
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self):
        port = self.cmb_port.currentText().strip()
        baud = int(self.cmb_baud.currentText())

        self.cfg.last_port = port
        self.cfg.baud = baud
        self.cfg.save()

        # Create worker and set port/baud
        self.worker = SerialWorker()
        self.worker.port = port if port else 'COM3'
        self.worker.baud = baud
        
        # Connect signals
        self.worker.status.connect(self.status.showMessage)
        self.worker.connection_status.connect(self.on_connection_status)
        self.worker.reading.connect(self.on_reading)
        self.worker.finished.connect(self.on_worker_finished)
        
        # Reset settling counter for sensor stabilization
        self._settling_samples = 0
        
        # Clear old data from previous session
        self._samples_for_bpm.clear()
        self._bpm_vals.clear()
        self.lbl_bpm.setText("--")
        self.update_bpm_stats()

        self._has_good_signal = False
        self._no_signal_samples = 0
        
        # Restart BPM timer if it was stopped
        if hasattr(self, '_bpm_timer') and self._bpm_timer:
            # if not self._bpm_timer.isActive():
            self._bpm_timer.start(1000)
        
        # Start worker using QTimer.singleShot to ensure Qt event loop is running
        QtCore.QTimer.singleShot(100, self.worker.run)
        
        self._connected = True
        self.btn_connect.setText("⏸ Disconnect")
        self.btn_connect.setStyleSheet("""
            QToolButton {
                background-color: #E74C3C;
            }
            QToolButton:hover {
                background-color: #C0392B;
            }
        """)
        self.status.showMessage(f"Connecting to {port} @ {baud} baud...")

    def disconnect_serial(self):
        self._connected = False
        
        # Stop BPM calculation timer
        if hasattr(self, '_bpm_timer') and self._bpm_timer:
            self._bpm_timer.stop()
        
        # Clear data buffers
        self._samples_for_bpm.clear()
        self._bpm_vals.clear()

        self._has_good_signal = False
        self._no_signal_samples = 0
        self._settling_samples = 0
        
        # Reset display
        self.lbl_bpm.setText("--")
        self.update_bpm_stats()
        
        if self.worker:
            try:
                self.worker.disconnect()
            except Exception as e:
                self.status.showMessage(f"Disconnect error: {e}")
            finally:
                self.worker = None
        
        self.btn_connect.setText("▶ Connect")
        self.btn_connect.setStyleSheet("""
            QToolButton {
                background-color: #4A90E2;
            }
            QToolButton:hover {
                background-color: #357ABD;
            }
        """)
        self.status.showMessage("Disconnected")
    
    def on_connection_status(self, msg: str):
        """Handle connection status updates from worker."""
        self.status.showMessage(msg)
        if "Connected" in msg:
            self._connected = True
        elif "Disconnected" in msg or "Error" in msg:
            self._connected = False

    def on_worker_finished(self):
        self._connected = False
        self.btn_connect.setText("▶ Connect")
        self.btn_connect.setStyleSheet("""
            QToolButton {
                background-color: #4A90E2;
            }
            QToolButton:hover {
                background-color: #357ABD;
            }
        """)
        self.status.showMessage("Stopped")

    # ----- Data handlers -----

    def on_reading(self, values: list):
        """Handle reading signal from worker - receives a list of PPG values."""
        if not values:
            return
        
        current_time = time.time()

        # Heuristic thresholds – adjust as needed for your sensor
        NOISE_THRESH = 100      # values <= this are considered "no signal / noise"
        MIN_GOOD_FRACTION = 0.3 # fraction of samples that must be > NOISE_THRESH to call this "good"
        NO_SIGNAL_RESET_SAMPLES = 50  # how many low samples before we declare signal lost
        SETTLING_REQUIRED_SAMPLES = 50  # samples to ignore after we first see good signal

        # --- 1. Classify this block of samples ---
        high_values = [v for v in values if v > NOISE_THRESH]
        low_values = [v for v in values if v <= NOISE_THRESH]

        if not high_values:
            # No good samples at all → likely no finger or transition
            self._no_signal_samples += len(values)
        else:
            # We saw some good amplitude; reset "no signal" counter
            self._no_signal_samples = 0

        # --- 2. If we had good signal and now lost it for a while → reset BPM state ---
        if self._has_good_signal and self._no_signal_samples > NO_SIGNAL_RESET_SAMPLES:
            # Treat as "finger removed / sensor reset"
            self._has_good_signal = False
            self._settling_samples = 0
            self._samples_for_bpm.clear()
            self._bpm_vals.clear()
            self.lbl_bpm.setText("--")
            self.update_bpm_stats()

        # --- 3. If we do NOT yet have good signal, check if this block starts one ---
        if not self._has_good_signal:
            # Enough high values in this chunk? Then we say "finger just placed"
            if len(high_values) >= max(5, int(MIN_GOOD_FRACTION * len(values))):
                self._has_good_signal = True
                self._settling_samples = 0      # start settling window from this point
                self._samples_for_bpm.clear()   # drop any garbage that came before

        # --- 4. Always update the waveform display (so user can see noise/flat line) ---
        for v in values:
            self.chart.append(float(v))

        # --- 5. If we still don't have good signal, don't drive BPM at all ---
        if not self._has_good_signal:
            # Still recording? We can still log raw samples, but no BPM
            if self.recording and self.csv_writer:
                ts = time.time()
                for v in values:
                    if v > 0:  # keep your existing "no zero" rule
                        self.csv_writer.writerow([ts, "S", f"{v:.3f}"])
            return

        # --- 6. We have good signal: respect settling window before BPM ---
        self._settling_samples += len(values)
        if self._settling_samples < SETTLING_REQUIRED_SAMPLES:
            # During settling, we show waveform but don't accumulate BPM data yet
            if self.recording and self.csv_writer:
                ts = time.time()
                for v in values:
                    if v > 0:
                        self.csv_writer.writerow([ts, "S", f"{v:.3f}"])
            return

        # --- 7. After settling: accumulate samples for BPM ---
        for v in values:
            # Only register samples that look reasonably real
            if v > NOISE_THRESH:
                self._samples_for_bpm.append((current_time, float(v)))

        # --- 8. Recording raw samples (same as before) ---
        if self.recording and self.csv_writer:
            ts = time.time()
            for v in values:
                if v > 0:  # Don't record zero values
                    self.csv_writer.writerow([ts, "S", f"{v:.3f}"])


    def calculate_bpm(self):
        """Calculate BPM from PPG waveform using peak detection."""
        if len(self._samples_for_bpm) < 50:
            # Not enough data yet
            return
        
        # Convert to numpy arrays for easier processing
        samples = list(self._samples_for_bpm)
        times = np.array([s[0] for s in samples])
        values = np.array([s[1] for s in samples])
        
        # Simple peak detection with adaptive threshold
        # Calculate moving average and standard deviation
        window_size = min(20, len(values) // 5)
        if window_size < 3:
            return
        
        # Use a simple threshold: mean + 0.5 * std
        mean_val = np.mean(values)
        std_val = np.std(values)
        threshold = mean_val + 0.5 * std_val
        
        # Find peaks: values above threshold with rising then falling edge
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > threshold:
                # Check if this is a local maximum
                if values[i] > values[i-1] and values[i] > values[i+1]:
                    peaks.append(times[i])
        
        # Need at least 2 peaks to calculate BPM
        if len(peaks) < 2:
            return
        
        # Calculate inter-beat intervals (IBI)
        ibis = []
        for i in range(1, len(peaks)):
            ibi = peaks[i] - peaks[i-1]
            # Filter out unrealistic IBIs (BPM between 30 and 200)
            if 0.3 < ibi < 2.0:  # 30 BPM to 200 BPM range
                ibis.append(ibi)
        
        if not ibis:
            return
        
        # Calculate BPM from average IBI
        avg_ibi = np.mean(ibis)
        bpm = 60.0 / avg_ibi
        
        # Update display
        self.lbl_bpm.setText(f"{bpm:.0f}")
        self._bpm_vals.append(bpm)
        
        # Update statistics
        self.update_bpm_stats()
        
        # Record BPM if recording
        if self.recording and self.csv_writer:
            ts = time.time()
            self.csv_writer.writerow([ts, "BPM", f"{bpm:.2f}"])
    def update_bpm_stats(self):
        """Update BPM statistics display."""
        if self._bpm_vals:
            arr = np.array(self._bpm_vals)
            self.lbl_bpm_min.setText(f"{arr.min():.0f}")
            self.lbl_bpm_max.setText(f"{arr.max():.0f}")
            self.lbl_bpm_avg.setText(f"{arr.mean():.0f}")
            # Simple quality indicator based on variance
            variance = np.var(arr)
            if variance < 25:  # Low variance = good quality
                self.lbl_quality.setText("Good")
                self.lbl_quality.setStyleSheet("font-size: 18px; font-weight: 600; color: #00A86B;")
            elif variance < 100:
                self.lbl_quality.setText("Fair")
                self.lbl_quality.setStyleSheet("font-size: 18px; font-weight: 600; color: #F39C12;")
            else:
                self.lbl_quality.setText("Poor")
                self.lbl_quality.setStyleSheet("font-size: 18px; font-weight: 600; color: #E74C3C;")
        else:
            self.lbl_bpm.setText("--")
            self.lbl_bpm_min.setText("--")
            self.lbl_bpm_max.setText("--")
            self.lbl_bpm_avg.setText("--")
            self.lbl_quality.setText("--")
            self.lbl_quality.setStyleSheet("font-size: 18px; font-weight: 600; color: #00A86B;")

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
        self.btn_record.setText("⏹ Stop" if on else "⏺ Record")
        self.recording_indicator.setVisible(on)
        if on:
            # Pulse animation could be added here for extra visibility
            self.recording_indicator.raise_()

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
