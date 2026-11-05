# Through-Wall Human Detection Using Wi-Fi CSI

Detects a person behind a wall using only Wi-Fi signals, with no camera and no line of sight. Built on a Seeed Studio XIAO ESP32-C6 with a Python signal-processing pipeline, a live dashboard, and LED and buzzer alerts.

Built as a team of three for an Electronics & Telecommunication mini-project (Savitribai Phule Pune University, 2025–26).

## How it works

Wi-Fi signals pass through walls and change slightly when a person moves in their path. The ESP32-C6 reads these changes as Channel State Information (CSI), which records the amplitude of 52 OFDM subcarriers.

1. **Capture:** The ESP32-C6 firmware connects to a Wi-Fi router, collects CSI from the router's traffic, and streams it over USB serial at 2 Mbaud.
2. **Clean:** A Hampel filter removes spikes and outliers.
3. **Remove static reflections:** Subtracting each subcarrier's mean removes the constant reflections from walls and furniture.
4. **Isolate motion:** PCA separates the moving components from the rest of the signal.
5. **Detect:** An adaptive detector tracks the quiet baseline over time and flags presence when motion rises clearly above it. No manual empty-room calibration is needed for each new location.
6. **Report:** The Python dashboard shows a CSI heatmap, FFT spectrum, presence status and live metrics. The hardware shows green (empty) or red (present) LEDs and sounds a buzzer on detection.

## Hardware

- Seeed Studio XIAO ESP32-C6
- External 2.4 GHz antenna (U.FL connector)
- 2 green and 2 red LEDs
- Passive buzzer driven by a BC547 transistor
- Wi-Fi router as the CSI source (tested on channel 6, 2.437 GHz)

Total cost: under Rs. 1,000.

Pin mapping (from the firmware):

| Pin | Function |
|-----|----------|
| D0, D7 | Green LEDs |
| D1, D6 | Red LEDs |
| D2 | Buzzer (via BC547) |

## Results

Tested with a router on channel 6 and a Python dashboard running on a laptop connected over USB.

| Test | Distance | Wall | Activity | Result |
|------|----------|------|----------|--------|
| TC-01 | 1 m | None | Walking | Detected |
| TC-02 | 2 m | None | Standing still | Partial |
| TC-03 | 1 m | 10 cm drywall | Walking | Detected |
| TC-04 | 2 m | 10 cm drywall | Standing still | Detected |
| TC-05 | 3 m | 10 cm drywall | Walking | Detected |
| TC-06 | 5 m | 10 cm drywall | Walking | **Missed** |
| TC-07 | 2 m | 20 cm brick | Walking | Detected |
| TC-08 | 2 m | 20 cm brick | Standing still | Partial |

Summary: 5 full detections, 2 partial (both standing still), 1 miss (walking through drywall at 5 m). Detection latency was under 500 ms in positive cases, and false positives were about 5%, mostly caused by air movement from HVAC.

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point: runs capture, processing, detection and the dashboard |
| `csi_receiver.py` | Reads CSI lines from the serial port |
| `hampel_filter.py` | Spike removal |
| `preprocessor.py` | Static removal and PCA |
| `detector.py` | Adaptive baseline and presence decision |
| `calibrate.py` | Optional threshold calibration |
| `dashboard.py` | Live heatmap, FFT and status display |
| `gpio_controller.py` | Sends LED and buzzer commands to the ESP32 |
| `config.py` | Serial port, baud rate, buffer sizes and thresholds |
| `test_runner.py` | Automated tests (TC-SW-01 to TC-SW-05) |
| `esp32c6_firmware/` | Arduino firmware for the ESP32-C6 |
| `data/` | Baseline and calibration logs |

## Setup

### 1. Flash the firmware

1. Open `esp32c6_firmware/esp32c6_firmware.ino` in the Arduino IDE.
2. Install the ESP32 board package and select the XIAO ESP32-C6.
3. Replace the Wi-Fi name and password in the file with your own network.
4. Upload the sketch.

Do not commit your Wi-Fi credentials to a public repository.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3 with numpy, scipy, matplotlib and pyserial.

### 3. Configure the serial port

In `config.py`, set `SERIAL_PORT` to your board's port:

- Windows: `COM3`, `COM7`, etc.
- Linux: `/dev/ttyACM0`
- macOS: `/dev/cu.usbserial-*`

You can also pass the port on the command line (see below).

### 4. Run

```bash
python main.py                  # use the port from config.py
python main.py --port COM7      # override the port
python main.py --no-gpio        # run without the LED/buzzer hardware
python main.py --threshold 3600 # override the detection threshold
```

### 5. Run the tests

```bash
python test_runner.py
```

Hardware tests are skipped automatically if the board is not connected.

## Limitations

- Works for walking through interior walls up to 20 cm thick (drywall and brick) at short range. Standing-still detection is less reliable.
- Detection through 10 cm drywall at 5 m failed in the test run.
- Tested with one router and one room. Results will vary with layout, router placement and interference.
- The detector is a threshold-based presence check, not a trained classifier.

## Future scope

- Lightweight machine learning (TinyML on the ESP32) to classify activities such as walking, sitting and falling
- Multiple ESP32-C6 nodes for location estimation within a room
- MQTT streaming of detection events to a cloud dashboard
- 5 GHz CSI for finer resolution
- Better denoising, such as wavelet or deep-learning methods

## Acknowledgements

Mini-project coordinator: Prof. R. K. Patil. Head of Department: Dr. P. G. Shete.

Team: Ashutosh Arvind Panchal, Omkar Krishna Muske, Avinash Dattatray Ganore.