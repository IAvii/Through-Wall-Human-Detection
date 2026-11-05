import os
import re
import sys
import time
import numpy as np
from datetime import datetime

from config import (
    SERIAL_PORT,
    SERIAL_BAUD,
    BUFFER_SIZE,
    SAMPLING_RATE,
    CALIBRATION_WINDOWS,
    BASELINE_FILE,
    CALIBRATION_LOG,
    DATA_DIR
)
from csi_receiver import CSIReceiver
from hampel_filter import hampel_filter_vectorized
from preprocessor import preprocess

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        WiFi Radar — EMPTY ROOM CALIBRATION             ║")
    print("║        Through-Wall Human Detection System              ║")
    print("║        XIAO ESP32-C6 CSI Collector                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

def print_instructions():
    print("=== CALIBRATION MODE ===")
    print()
    print("IMPORTANT: Follow these steps carefully:")
    print()
    print("  1. Ensure Room B (behind the wall) is completely EMPTY.")
    print("     → No person, no pet, no moving objects.")
    print("  2. Keep the WiFi router powered ON and connected.")
    print("  3. Do NOT walk near the antenna during calibration.")
    print("  4. The ESP32-C6 must be running and streaming CSI data.")
    print()
    print(f"  Collecting {CALIBRATION_WINDOWS} windows of data")
    print(f"  ({CALIBRATION_WINDOWS * BUFFER_SIZE / SAMPLING_RATE:.0f} seconds approximate)")
    print()

def update_config_threshold(new_threshold):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # replace the detection_threshold line
        pattern = r'(DETECTION_THRESHOLD\s*=\s*)[\d.]+'
        replacement = f'\\g<1>{new_threshold:.6f}'

        new_content, count = re.subn(pattern, replacement, content)

        if count == 0:
            print("[WARNING] Could not find DETECTION_THRESHOLD in config.py")
            print(f"  Please manually set: DETECTION_THRESHOLD = {new_threshold:.6f}")
            return False

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True

    except (IOError, OSError) as e:
        print(f"[ERROR] Could not update config.py: {e}")
        return False

def save_calibration_log(threshold, variances):
    os.makedirs(DATA_DIR, exist_ok=True)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CALIBRATION_LOG)

    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n{'='*60}\n")
            f.write(f"Calibration: {timestamp}\n")
            f.write(f"Threshold:   {threshold:.6f}\n")
            f.write(f"Variances:   {[f'{v:.4f}' for v in variances]}\n")
            f.write(f"Mean:        {np.mean(variances):.6f}\n")
            f.write(f"Std:         {np.std(variances):.6f}\n")
            f.write(f"{'='*60}\n")
        print(f"  Calibration log saved to {CALIBRATION_LOG}")
    except (IOError, OSError) as e:
        print(f"  [WARNING] Could not save calibration log: {e}")

def run_calibration():
    print_banner()
    print_instructions()

    # prompt user confirmation
    print("─" * 58)
    input("  Press ENTER when Room B is empty and you are ready... ")
    print("─" * 58)
    print()

    # connect to esp32-c6
    print(f"Connecting to ESP32-C6 on {SERIAL_PORT}...")
    try:
        receiver = CSIReceiver(port=SERIAL_PORT, baud=SERIAL_BAUD, buffer_size=BUFFER_SIZE)
    except Exception as e:
        print(f"\n[ERROR] Could not connect to serial port: {e}")
        print(f"  Check that ESP32-C6 is connected and SERIAL_PORT in config.py is correct.")
        print(f"  Current setting: SERIAL_PORT = \"{SERIAL_PORT}\"")
        sys.exit(1)

    # wait for buffer to fill initially
    print("Waiting for CSI data buffer to fill...")
    timeout = 30
    start = time.time()
    while receiver.buffer_fill < BUFFER_SIZE:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\n[ERROR] Timeout — no CSI data received in {timeout} seconds.")
            print("  Ensure ESP32-C6 firmware is running and WiFi is connected.")
            receiver.close()
            sys.exit(1)
        fill = receiver.buffer_fill
        pct = int(fill / BUFFER_SIZE * 100)
        bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
        print(f"\r  Buffer fill: [{bar}] {fill}/{BUFFER_SIZE} ({pct}%)", end='', flush=True)
        time.sleep(0.5)
    print(f"\r  Buffer fill: [{'█' * 20}] {BUFFER_SIZE}/{BUFFER_SIZE} (100%)")
    print()

    # collect calibration windows
    signal_variances = []
    explained_variances = []
    outlier_counts = []

    print("Collecting calibration data...")
    print()

    try:
        for window_idx in range(CALIBRATION_WINDOWS):
            # wait for fresh data (step_size worth of new packets)
            time.sleep(BUFFER_SIZE / SAMPLING_RATE * 0.2)  # wait ~2 seconds between windows

            matrix = receiver.get_matrix()
            if matrix is None:
                print(f"  Window {window_idx + 1}/{CALIBRATION_WINDOWS}: "
                      f"Buffer not ready — retrying...")
                time.sleep(2.0)
                matrix = receiver.get_matrix()
                if matrix is None:
                    print("  [WARNING] Skipping window — no data")
                    continue

            # run processing pipeline
            cleaned, n_outliers = hampel_filter_vectorized(matrix)
            dominant_signal, explained_var = preprocess(cleaned)

            # compute variance of the dominant signal
            sig_variance = float(np.var(dominant_signal))

            signal_variances.append(sig_variance)
            explained_variances.append(explained_var)
            outlier_counts.append(n_outliers)

            # progress display
            bar_len = 20
            done = int((window_idx + 1) / CALIBRATION_WINDOWS * bar_len)
            bar = '█' * done + '░' * (bar_len - done)
            print(f"  Window {window_idx + 1:2d}/{CALIBRATION_WINDOWS} [{bar}] "
                  f"variance={sig_variance:.4f}  pca_var={explained_var:.1f}%  "
                  f"outliers={n_outliers}")

        print()

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Calibration cancelled by user.")
        receiver.close()
        sys.exit(0)

    # validate we have enough data
    if len(signal_variances) < 3:
        print("[ERROR] Too few valid windows collected. Try again.")
        receiver.close()
        sys.exit(1)

    # compute threshold: mean + 3*std of empty room variances
    var_array = np.array(signal_variances)
    mean_var = np.mean(var_array)
    std_var = np.std(var_array)
    threshold = mean_var + 3.0 * std_var

    # ensure minimum threshold
    threshold = max(threshold, 1.0)

    # display results
    print("═" * 58)
    print("  CALIBRATION RESULTS")
    print("═" * 58)
    print(f"  Windows collected:     {len(signal_variances)}")
    print(f"  Signal variances:      {[f'{v:.4f}' for v in signal_variances]}")
    print(f"  Mean variance:         {mean_var:.6f}")
    print(f"  Std variance:          {std_var:.6f}")
    print(f"  Mean explained var:    {np.mean(explained_variances):.2f}%")
    print(f"  Mean outliers/window:  {np.mean(outlier_counts):.0f}")
    print()
    print(f"  ► Computed threshold:  {threshold:.6f}")
    print(f"    (mean + 3σ = {mean_var:.6f} + 3 × {std_var:.6f})")
    print("═" * 58)
    print()

    # save threshold to config.py
    print("Saving threshold to config.py...")
    if update_config_threshold(threshold):
        print(f"  ✓ DETECTION_THRESHOLD = {threshold:.6f} saved to config.py")
    else:
        print(f"  ✗ Manual update required: DETECTION_THRESHOLD = {threshold:.6f}")

    # save baseline data
    baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), BASELINE_FILE)
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)

    try:
        np.save(baseline_path, var_array)
        print(f"  ✓ Baseline data saved to {BASELINE_FILE}")
    except (IOError, OSError) as e:
        print(f"  ✗ Could not save baseline: {e}")

    # save calibration log
    save_calibration_log(threshold, signal_variances)

    # cleanup
    receiver.close()

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CALIBRATION COMPLETE                                    ║")
    print("║  You can now run: python main.py                         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

if __name__ == "__main__":
    run_calibration()
