from collections import deque
import serial
import time
from PySide6.QtCore import QObject, QTimer, Signal
from collections import deque




class SerialWorker(QObject):
    status = Signal(str)
    connection_status = Signal(str)
    reading = Signal(deque)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.port = 'COM3'
        self.baud = 9600
        self.ser = None
        self.buffer = deque(maxlen=12)
        self._connected = False
        self._running = True
        self.run_timer = None
        self.emit_timer = None
    
    def connect(self, port, baud):
        self.status.emit(f"Attempting to connect to Port: {port} at baud {baud}")
        try: 
            self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.2) #timeout karg makes thread non-blocking
            time.sleep(1)
            self.ser.reset_input_buffer()
            self.status.emit(f"Successfully Connected to Port: {port} at baud {baud}")
            self._connected = True
            return 0
        except Exception as e:
            self.status.emit(f"Connection Error: {e}")
            self._connected = False
            return -1
        
    # def refreshPorts(self):
    #     self.list_ports = serial.tools.list_ports()
    #     return (self.list_ports)
    
    def disconnect(self):
        self.status.emit("Disconnecting...")
        self._running = False
        self._connected = False
        
        # Stop timers first
        if self.run_timer and self.run_timer.isActive():
            self.run_timer.stop()
        if self.emit_timer and self.emit_timer.isActive():
            self.emit_timer.stop()
        
        # Close serial connection
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        self.status.emit("Disconnected")
        self.finished.emit()
        
        
    def fillBuffer(self):
        if not (self._running and self._connected):
            return
        for _ in range(3):
            try:
                raw = self.ser.readline().decode('utf-8').strip()
                if not raw:
                    continue
                lower = raw.lower()
                if lower.startswith('signal'):
                    raw = raw.replace("Signal", "").replace(":", "").strip()
                value = int(raw)
                self.buffer.append(value)
            except ValueError:
                pass
    
    def copyAndClearBuffer(self):
        if self.buffer:
            buf_copy = list(self.buffer)
            self.reading.emit(buf_copy)
            self.buffer.clear()
        
    def run(self):
        self.run_timer = QTimer(self)
        self.run_timer.timeout.connect(self.fillBuffer)
        self.run_timer.start(10)
       
        self.emit_timer = QTimer(self)
        self.emit_timer.timeout.connect(self.copyAndClearBuffer)
        self.emit_timer.start(100)
        
        if self._connected is False:
            con_status = self.connect(self.port, self.baud)
            if con_status == 0:
                self.connection_status.emit("Connected")
                self.connection_status.emit("Reading Pulse")
            else:
                self.status.emit("Connection Error in run()")
        










