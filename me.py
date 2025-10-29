from PySide6.QtWidgets import (QApplication, 
                               QMainWindow, 
                               QWidget, 
                               QHBoxLayout,
                               QToolBar, 
                               QPushButton,
                               QLabel,
                               )
from PySide6.QtCore import QThread
import pyqtgraph as pg
from worker import SerialWorker


class PPGWaveformPlot(pg.PlotWidget):
    def __init__(self):
        super().__init__(background='w')
        self.setTitle("PPG Waveform", color="#333", size="12pt")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel("left", "Amplitude (arb.)")
        self.setLabel("bottom", "Time (s)")
        

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Rate Project")
        self.resize(1100, 680)

        # Main Areas
        central = QWidget(self)
        self.setCentralWidget(central)

        primary_widgets = QHBoxLayout(central)
        self.tb = QToolBar()

        
        self.plot = PPGWaveformPlot()
        

        self.addToolBar(self.tb)


        primary_widgets.addWidget(self.plot)
        # Toolbar Components
        self.connect_btn = QPushButton("Connect")
        self._connected = False
        self.connect_btn.clicked.connect(self._connect_btn_toggle)
        status_label = QLabel("Status: ")
        self.status_update = QLabel()

        temp_reading = QLabel("Reading: ")
        self.reading = QLabel()

        # Integrate Toolbar components
        self.tb.addWidget(self.connect_btn)
        self.tb.addWidget(status_label)
        self.tb.addWidget(self.status_update)
        self.tb.addWidget(temp_reading)
        self.tb.addWidget(self.reading)

    def report_status(self, status):
        self.status_update.setText(status)

    def report_reading(self, _reading):
        self.reading.setText(str(_reading))
    
    def activateWorker(self):
        self.thread = QThread()
        self.worker = SerialWorker()
        self.worker.moveToThread(self.thread)
        self.worker.status.connect(self.report_status)
        self.worker.reading.connect(self.report_reading)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    # GUI states
    def _connect_btn_toggle(self):
        if not self._connected:
            self.connect_btn.setText("Disconnect")
            self._connected = True
            self.activateWorker()
        else:
            self.worker.disconnect()
            self.thread.quit()
            self.worker.deleteLater()
            self.thread.deleteLater()
            self.connect_btn.setText("Connect")
            self.status_update.setText("")
            self._connected = False


        




def main():
    app = QApplication()
    win = MainWindow()
    win.show()
    app.exec()

main()