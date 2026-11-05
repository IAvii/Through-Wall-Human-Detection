import os

# serial communication
SERIAL_PORT = "COM7"           # windows: "com3", "com4", etc.
                                # linux:   "/dev/ttyusb0" or "/dev/ttyacm0"
                                # macos:   "/dev/cu.usbserial-*"
SERIAL_BAUD = 2000000          # must match esp32-c6 firmware baud rate  # changed: was 921600

# buffer & sampling
BUFFER_SIZE = 500              # 500 packets = 10 seconds at 50 hz
STEP_SIZE = 50                 # slide 50 packets = 1 second between analyses
SAMPLING_RATE = 100            # expected csi packet rate in hz  # changed: was 50 — esp-now sends at 100 hz

# hampel filter
HAMPEL_WINDOW = 5              # half-window size k (full window = 2k+1 = 11)
HAMPEL_THRESHOLD = 3.0         # outlier threshold in mad units

# motion / presence detection
DETECTION_THRESHOLD = 3624.250260      # snr threshold: ratio of motion-band to noise-band energy

# consecutive detection
MIN_CONSECUTIVE_DETECTIONS = 10  # higher count for stable "human present" status

# gpio serial commands
GPIO_GREEN = 'G'               # green leds on (gpio2 + gpio9)
GPIO_RED = 'R'                 # red leds on (gpio3 + gpio8)
GPIO_BUZZ = 'B'                # buzzer pulse 500ms (gpio4)
GPIO_RESET = 'X'               # all gpio low

# file paths
DATA_DIR = "data/"
BASELINE_FILE = os.path.join("data", "empty_room_baseline.npy")
TRIAL_LOG = os.path.join("data", "trial_log.csv")
CALIBRATION_LOG = os.path.join("data", "calibration_log.txt")

# calibration
CALIBRATION_WINDOWS = 10       # number of data windows to collect during calibration

# ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
