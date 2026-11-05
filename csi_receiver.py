import time
import threading
import numpy as np
from collections import deque

import serial
import serial.tools.list_ports

from config import SERIAL_PORT, SERIAL_BAUD, BUFFER_SIZE

class CSIReceiver:
    
    def __init__(self, port=SERIAL_PORT, baud=SERIAL_BAUD, buffer_size=BUFFER_SIZE):
        self.port = port
        self.baud = baud
        self.buffer_size = buffer_size

        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._ser = None
        self._thread = None

        # packet rate tracking
        self._packet_timestamps = deque(maxlen=200)
        self._total_packets = 0
        self._parse_errors = 0

        # connect to serial port with retry logic
        self._connect(max_retries=3)

        # start reader thread
        self._running = True
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

    def _connect(self, max_retries=3):
        for attempt in range(1, max_retries + 1):
            try:
                self._ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baud,
                    timeout=1.0,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                # flush any stale data
                self._ser.reset_input_buffer()
                print(f"[CSIReceiver] Connected to {self.port} at {self.baud} baud")
                return
            except serial.SerialException as e:
                print(f"[CSIReceiver] Connection attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(1.0)
                else:
                    # list available ports for user reference
                    available = [p.device for p in serial.tools.list_ports.comports()]
                    print(f"[CSIReceiver] Available ports: {available}")
                    raise serial.SerialException(
                        f"Could not open {self.port} after {max_retries} attempts. "
                        f"Available ports: {available}. "
                        f"Update SERIAL_PORT in config.py."
                    )

    def _reader_thread(self):
        while self._running:
            try:
                if self._ser is None or not self._ser.is_open:
                    time.sleep(0.1)
                    continue

                raw = self._ser.readline()
                if not raw:
                    continue

                try:
                    line = raw.decode('utf-8', errors='ignore').strip()
                except UnicodeDecodeError:
                    continue

                if line:
                    self._parse_line(line)

            except serial.SerialException as e:
                print(f"[CSIReceiver] Serial error: {e}")
                time.sleep(0.5)
                # attempt reconnection
                try:
                    if self._ser:
                        self._ser.close()
                    self._connect(max_retries=1)
                except serial.SerialException:
                    pass
            except Exception as e:
                print(f"[CSIReceiver] Unexpected error in reader thread: {e}")
                time.sleep(0.1)

    def _parse_line(self, line):
        # skip status/debug lines from firmware — print them for diagnostics
        if any(tag in line for tag in ['READY', 'WIFI', '===', '[CSI]', '[NVS]',
                                        '[PING]', '[DEBUG]', '[ESP', '[NETIF]']):
            print(f"  [ESP32] {line}")
            return
        # skip non-data lines (status messages, debug output)
        if not line[0].isdigit() and line[0] != '-':
            if len(line) > 2:
                print(f"  [ESP32] {line}")
            return

        try:
            parts = line.split(',')

            # minimum fields: timestamp, rssi, n_subcarriers, at least 1 amplitude
            if len(parts) < 4:
                self._parse_errors += 1
                return

            # extract header fields
            # timestamp_ms = int(parts[0])  # available but not stored
            # rssi = int(parts[1])          # available but not stored
            n_subcarriers = int(parts[2])

            # extract amplitude values (everything after the header)
            amp_values = parts[3:]

            # validate count matches
            if len(amp_values) < n_subcarriers:
                n_subcarriers = len(amp_values)

            if n_subcarriers == 0:
                return

            # convert to numpy float32 array
            amplitudes = np.array([float(a) for a in amp_values[:n_subcarriers]], dtype=np.float32)

            # thread-safe append to ring buffer
            with self._lock:
                self._buffer.append(amplitudes)
                self._packet_timestamps.append(time.time())
                self._total_packets += 1

        except (ValueError, IndexError) as e:
            self._parse_errors += 1

    def get_matrix(self):
        with self._lock:
            if len(self._buffer) < self.buffer_size:
                return None

            # all rows must have the same number of columns
            # use the most common column count
            col_counts = [len(row) for row in self._buffer]
            target_cols = max(set(col_counts), key=col_counts.count)

            # filter rows with matching column count
            valid_rows = [row for row in self._buffer if len(row) == target_cols]

            if len(valid_rows) < self.buffer_size:
                # pad with the last valid row if needed
                while len(valid_rows) < self.buffer_size:
                    valid_rows.append(valid_rows[-1].copy())

            return np.array(valid_rows[-self.buffer_size:], dtype=np.float32)

    def send_gpio(self, cmd):
        with self._lock:
            try:
                if self._ser and self._ser.is_open:
                    self._ser.write((cmd + '\n').encode('utf-8'))
                    self._ser.flush()
            except serial.SerialException as e:
                print(f"[CSIReceiver] GPIO command failed: {e}")

    @property
    def packet_rate(self):
        now = time.time()
        with self._lock:
            count = sum(1 for ts in self._packet_timestamps if now - ts <= 1.0)
        return count

    @property
    def buffer_fill(self):
        with self._lock:
            return len(self._buffer)

    @property
    def total_packets(self):
        return self._total_packets

    @property
    def parse_errors(self):
        return self._parse_errors

    def close(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
                print("[CSIReceiver] Serial port closed")
            except serial.SerialException:
                pass

if __name__ == "__main__":
    # quick test: connect and print packet rate
    print("Testing CSI Receiver...")
    try:
        receiver = CSIReceiver()
        print(f"Listening on {SERIAL_PORT} at {SERIAL_BAUD} baud...")
        print("Press Ctrl+C to stop\n")

        while True:
            time.sleep(1.0)
            rate = receiver.packet_rate
            fill = receiver.buffer_fill
            matrix = receiver.get_matrix()
            status = "READY" if matrix is not None else "FILLING"
            print(f"  Packets/sec: {rate:3d}  |  Buffer: {fill}/{BUFFER_SIZE}  |  {status}")

    except serial.SerialException as e:
        print(f"\nError: {e}")
        print("Check that the ESP32-C6 is connected and SERIAL_PORT in config.py is correct.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'receiver' in locals():
            receiver.close()
