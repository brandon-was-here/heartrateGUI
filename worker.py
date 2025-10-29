import serial
import time
from PySide6.QtCore import QObject, QThread, Signal
from collections import deque




class SerialWorker(QObject):
    port = 'COM3'
    baud = 9600
    ser = None
    status = Signal(str)
    connection_status = Signal(str)
    reading = Signal(int)
    stop = Signal()
    _connected = False
    _running = True
    
    def connect(self, port, baud):
        self.status.emit(f"Attempting to connect to Port: {port} at baud {baud}")
        try: 
            self.ser = serial.Serial(port=port, baudrate=baud)
            time.sleep(1)
            self.ser.reset_input_buffer()
            self.status.emit(f"Successfully Connected to Port: {port} at baud {baud}")
            self._connected = True
            return 0
        except Exception as e:
            self.status.emit(f"Connection Error: {e}")
            self._connected = False
            return -1
    
    def disconnect(self):
        self.status.emit("Disconnecting...")
        self._running = False
        self._connected = False
        self.status.emit("Disconnected")
        
        
    def readSignal(self):
        while self._connected:
            try:
                raw = self.ser.readline().decode('utf-8').strip()
                if not raw:
                    continue
                lower = raw.lower()
                if lower.startswith('signal'):
                    raw = raw.replace("Signal", "").replace(":", "").strip()
                value = int(raw)
                yield value
            except ValueError:
                continue
        
    def run(self):
        if self._connected is False:
            con_status = self.connect(self.port, self.baud)
            if con_status == 0:
                self.connection_status.emit("Connected")
                self.connection_status.emit("Reading Pulse")
                for v in self.readSignal():
                    if not self._running:
                        break
                    self.reading.emit(v)
            else:
                self.status.emit("Connection Error in run()")
        










