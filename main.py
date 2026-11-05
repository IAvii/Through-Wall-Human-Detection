import os
import sys
import time
import argparse
import atexit
from datetime import datetime

import matplotlib.pyplot as plt

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║     ██╗    ██╗██╗███████╗██╗    ██████╗  █████╗ ██████╗      ║")
    print("║     ██║    ██║██║██╔════╝██║    ██╔══██╗██╔══██╗██╔══██╗     ║")
    print("║     ██║ █╗ ██║██║█████╗  ██║    ██████╔╝███████║██║  ██║     ║")
    print("║     ██║███╗██║██║██╔══╝  ██║    ██╔══██╗██╔══██║██║  ██║     ║")
    print("║     ╚███╔███╔╝██║██║     ██║    ██║  ██║██║  ██║██████╔╝     ║")
    print("║      ╚══╝╚══╝ ╚═╝╚═╝     ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝      ║")
    print("║                                                              ║")
    print("║     Through-Wall Human Detection Using WiFi CSI Signals      ║")
    print("║     Hardware: Seeed Studio XIAO ESP32-C6                     ║")
    print("║     Pipeline: Hampel → PCA → Variance → Detection            ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='WiFi Radar — Through-Wall Human Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--port', type=str, default=None,
                        help='Serial port (overrides config.py SERIAL_PORT)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Detection threshold (overrides config.py DETECTION_THRESHOLD)')
    parser.add_argument('--no-gpio', action='store_true',
                        help='Disable GPIO commands (for testing without hardware)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    print_banner()

    # import after banner so user sees something immediately
    from config import (
        SERIAL_PORT, SERIAL_BAUD, BUFFER_SIZE, SAMPLING_RATE,
        DETECTION_THRESHOLD, TRIAL_LOG, DATA_DIR
    )
    from csi_receiver import CSIReceiver
    from gpio_controller import GPIOController
    from dashboard import create_dashboard
    from detector import detect

    # apply overrides
    port = args.port if args.port else SERIAL_PORT
    threshold = args.threshold if args.threshold else DETECTION_THRESHOLD
    gpio_enabled = not args.no_gpio

    # print configuration
    print("  Configuration:")
    print(f"    Serial port:      {port}")
    print(f"    Baud rate:        {SERIAL_BAUD}")
    print(f"    Buffer size:      {BUFFER_SIZE} packets")
    print(f"    Sampling rate:    {SAMPLING_RATE} Hz")
    print(f"    Threshold:        {threshold}")
    print(f"    GPIO enabled:     {gpio_enabled}")
    print()

    # connect to esp32-c6
    print(f"  Connecting to ESP32-C6 on {port}...")
    try:
        receiver = CSIReceiver(port=port, baud=SERIAL_BAUD, buffer_size=BUFFER_SIZE)
    except Exception as e:
        print(f"\n  ✗ ERROR: Could not open serial port {port}")
        print(f"    {e}")
        print()
        print("  Troubleshooting:")
        print(f"    1. Check that ESP32-C6 is connected via USB-C data cable")
        print(f"    2. Open Device Manager → Ports (COM & LPT) → find correct port")
        print(f"    3. Update SERIAL_PORT in config.py or use --port flag")
        print(f"    4. Close Arduino Serial Monitor if open (it locks the port)")
        print()
        print(f"  Current setting: SERIAL_PORT = \"{port}\"")
        sys.exit(1)

    # wait for first packets
    print("  Waiting for CSI packets", end='', flush=True)
    timeout = 10
    start = time.time()
    while receiver.buffer_fill == 0:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\n\n  ✗ ERROR: No CSI packets received in {timeout} seconds.")
            print()
            print("  Troubleshooting:")
            print("    1. Open Arduino IDE Serial Monitor at 921600 baud")
            print("       → You should see CSV lines like: 1234567,-45,52,12,15,...")
            print("    2. If no output: re-upload firmware to ESP32-C6")
            print("    3. Check that WiFi router is ON and ESP32 connected to it")
            print("    4. Ensure 2.4 GHz antenna is attached to U.FL connector")
            print("    5. Close Serial Monitor before running this script")
            receiver.close()
            sys.exit(1)
        print('.', end='', flush=True)
        time.sleep(0.5)

    print(f" ✓ (first packet in {time.time() - start:.1f}s)")
    print(f"  Packet rate: {receiver.packet_rate} pkt/s")
    print()

    # wait for buffer to fill
    print("  Filling buffer", end='', flush=True)
    while receiver.buffer_fill < BUFFER_SIZE:
        fill_pct = int(receiver.buffer_fill / BUFFER_SIZE * 100)
        print(f"\r  Filling buffer... {receiver.buffer_fill}/{BUFFER_SIZE} ({fill_pct}%)", end='', flush=True)
        time.sleep(0.5)
    print(f"\r  Filling buffer... {BUFFER_SIZE}/{BUFFER_SIZE} (100%) ✓")
    print()

    # create gpio controller
    gpio_ctrl = GPIOController(receiver, enabled=gpio_enabled)

    # send red led at startup (system starting)
    if gpio_enabled:
        gpio_ctrl.red_on()
        print("  ● Red LED ON — system starting up")

    # session tracking
    session_start = datetime.now()

    def save_session_log():
        try:
            session_end = datetime.now()
            duration = (session_end - session_start).total_seconds()
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRIAL_LOG)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            # create header if file doesn't exist
            write_header = not os.path.exists(log_path)

            with open(log_path, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write("timestamp,duration_s,total_windows,detections,empty,threshold\n")
                f.write(f"{session_start.strftime('%Y-%m-%d %H:%M:%S')},"
                        f"{duration:.0f},{state.get('total_windows', 0)},"
                        f"{state.get('detection_count', 0)},"
                        f"{state.get('empty_count', 0)},{threshold}\n")

            print(f"  Session log saved to {TRIAL_LOG}")
        except Exception as e:
            print(f"  [WARNING] Could not save session log: {e}")

    atexit.register(save_session_log)

    # launch dashboard
    print("  Launching dashboard...")
    print()
    print("─" * 62)
    print("  Dashboard is running. Close the window or press Ctrl+C to stop.")
    print("─" * 62)
    print()

    try:
        fig, ani, state = create_dashboard(
            receiver,
            detector_fn=None,  # use default detector
            gpio_ctrl=gpio_ctrl,
            threshold=threshold
        )
        plt.show()

    except KeyboardInterrupt:
        print("\n\n  Shutting down...")

    except Exception as e:
        print(f"\n  ✗ Dashboard error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # clean shutdown
        print()
        print("  Cleaning up...")

        # reset all gpio
        if gpio_enabled:
            try:
                receiver.send_gpio('X')
                print("  ● All GPIO reset (LEDs + buzzer OFF)")
            except Exception:
                pass

        # close serial connection
        receiver.close()

        # print session summary
        print()
        print("═" * 62)
        print("  SESSION SUMMARY")
        print("═" * 62)
        print(f"  Duration:     {(datetime.now() - session_start).total_seconds():.0f} seconds")
        print(f"  Windows:      {state.get('total_windows', 0)}")
        print(f"  Detections:   {state.get('detection_count', 0)}")
        print(f"  Empty:        {state.get('empty_count', 0)}")
        total = state.get('detection_count', 0) + state.get('empty_count', 0)
        if total > 0:
            det_pct = state.get('detection_count', 0) / total * 100
            print(f"  Detection %:  {det_pct:.1f}%")
        print(f"  Packets:      {receiver.total_packets}")
        print("═" * 62)
        print()
        print("  WiFi Radar session ended. Goodbye!")
        print()

if __name__ == "__main__":
    main()
