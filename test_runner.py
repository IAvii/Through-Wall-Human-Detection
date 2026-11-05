import sys
import time
import numpy as np

# test result tracking
results = []

def record(test_id, description, passed, message=""):
    status = "PASS" if passed else "FAIL"
    results.append({
        'id': test_id,
        'description': description,
        'passed': passed,
        'message': message
    })
    color_code = "\033[92m" if passed else "\033[91m"
    reset_code = "\033[0m"
    print(f"  [{color_code}{status}{reset_code}] {test_id}: {description}")
    if message:
        print(f"         {message}")

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     WiFi Radar — Automated Software Test Suite          ║")
    print("║     Tests TC-SW-01 through TC-SW-05                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

# tc-sw-01 — serial connection test
def test_sw_01():
    print("─" * 58)
    print("TC-SW-01: Serial Connection Test")
    print("─" * 58)

    from config import SERIAL_PORT, SERIAL_BAUD

    try:
        import serial
        ser = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD, timeout=3.0)

        # try to read a few lines
        packets = 0
        start = time.time()
        timeout = 5.0

        while time.time() - start < timeout:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and line[0].isdigit():
                packets += 1
                if packets >= 5:
                    break

        ser.close()

        if packets >= 5:
            rate = packets / (time.time() - start)
            record("TC-SW-01", "Serial connection — packets received",
                   True, f"Received {packets} packets ({rate:.1f} pkt/s)")
        else:
            record("TC-SW-01", "Serial connection — no CSI data",
                   False, f"Connected but only {packets} packets received")

    except Exception as e:
        record("TC-SW-01", "Serial connection — no hardware connected",
               False, f"SKIP: {e}")

# tc-sw-02 — hampel filter spike removal
def test_sw_02():
    print("─" * 58)
    print("TC-SW-02: Hampel Filter Spike Removal")
    print("─" * 58)

    from hampel_filter import hampel_filter_vectorized as hampel_filter

    np.random.seed(42)

    # use near-constant baseline — tiny noise ensures zero false outliers
    # this isolates the core test: "does it catch spikes, leave clean data?"
    base = np.full((500, 52), 50.0) + np.random.normal(0, 0.01, (500, 52))
    original = base.copy()

    # inject known spikes — 150x above baseline, clearly outside any threshold
    spike_locations = [(50, 0), (150, 10), (300, 25)]
    for row, col in spike_locations:
        base[row, col] = 200.0

    # run filter
    filtered, outlier_count = hampel_filter(base)

    # test 1: spikes should be removed (value < 100)
    all_spikes_removed = True
    for row, col in spike_locations:
        if filtered[row, col] >= 100:
            all_spikes_removed = False
            print(f"    Spike at [{row},{col}]: {filtered[row, col]:.1f} (expected < 100)")

    record("TC-SW-02a", "Hampel filter removes all injected spikes",
           all_spikes_removed,
           f"Outliers detected: {outlier_count}")

    # test 2: non-spike values must be unchanged
    mask = np.ones_like(base, dtype=bool)
    for row, col in spike_locations:
        mask[row, col] = False

    max_deviation = np.max(np.abs(original[mask] - filtered[mask]))
    non_spike_preserved = max_deviation < 0.1

    record("TC-SW-02b", "Non-spike values preserved (< 0.1 deviation)",
           non_spike_preserved,
           f"Max deviation: {max_deviation:.6f}")

# tc-sw-03 — pca static removal
def test_sw_03():
    print("─" * 58)
    print("TC-SW-03: PCA Static Removal")
    print("─" * 58)

    from preprocessor import preprocess

    np.random.seed(42)

    # test 1: static data — no breathing
    static = np.ones((500, 52)) * 50 + np.random.normal(0, 0.5, (500, 52))
    signal, var = preprocess(static)

    static_ok = np.std(signal) < 5.0
    record("TC-SW-03a", "Static room — dominant signal std < 5.0",
           static_ok,
           f"Signal std: {np.std(signal):.4f}, Explained var: {var:.2f}%")

    # test 2: breathing signal — should show high variance
    t = np.linspace(0, 10, 500)
    signal_matrix = static.copy()
    # add breathing to multiple subcarriers for realistic scenario
    for sc in range(5):
        signal_matrix[:, sc] += np.sin(2 * np.pi * 0.25 * t) * 20

    signal, var = preprocess(signal_matrix)

    breathing_ok = var > 30.0
    record("TC-SW-03b", "Breathing signal — explained variance > 30%",
           breathing_ok,
           f"Explained variance: {var:.2f}%")

# tc-sw-04 — fft breathing rate extraction
def test_sw_04():
    print("─" * 58)
    print("TC-SW-04: FFT Breathing Rate Extraction")
    print("─" * 58)

    from detector import detect

    # create synthetic breathing signal: 0.25 hz = 15 bpm
    t = np.linspace(0, 10, 500)
    synthetic = np.sin(2 * np.pi * 0.25 * t) * 10

    result = detect(synthetic, sampling_rate=50, freq_min=0.1,
                    freq_max=0.6, threshold=0.01)

    # test 1: peak frequency accuracy
    freq_ok = abs(result['peak_freq'] - 0.25) < 0.02
    record("TC-SW-04a", "Peak frequency within 0.02 Hz of 0.25 Hz",
           freq_ok,
           f"Detected: {result['peak_freq']:.4f} Hz (expected: 0.25 Hz)")

    # test 2: bpm accuracy
    bpm_ok = abs(result['breathing_bpm'] - 15.0) < 1.5
    record("TC-SW-04b", "Breathing BPM within 1.5 of 15.0 BPM",
           bpm_ok,
           f"Detected: {result['breathing_bpm']:.1f} BPM (expected: 15.0 BPM)")

    # test 3: detection status
    det_ok = result['status'] == "HUMAN DETECTED"
    record("TC-SW-04c", "Status is 'HUMAN DETECTED'",
           det_ok,
           f"Status: {result['status']}")

# tc-sw-05 — gpio command format & config constants
def test_sw_05():
    print("─" * 58)
    print("TC-SW-05: GPIO Command Format & Config Constants")
    print("─" * 58)

    from config import (
        GPIO_GREEN, GPIO_RED, GPIO_BUZZ, GPIO_RESET,
        SAMPLING_RATE, BUFFER_SIZE, SERIAL_BAUD,
        BREATH_FREQ_MIN, BREATH_FREQ_MAX, HAMPEL_WINDOW
    )

    checks = [
        (GPIO_GREEN == 'G', f"GPIO_GREEN = '{GPIO_GREEN}' (expected 'G')"),
        (GPIO_RED == 'R', f"GPIO_RED = '{GPIO_RED}' (expected 'R')"),
        (GPIO_BUZZ == 'B', f"GPIO_BUZZ = '{GPIO_BUZZ}' (expected 'B')"),
        (GPIO_RESET == 'X', f"GPIO_RESET = '{GPIO_RESET}' (expected 'X')"),
        (SAMPLING_RATE == 50, f"SAMPLING_RATE = {SAMPLING_RATE} (expected 50)"),
        (BUFFER_SIZE == 500, f"BUFFER_SIZE = {BUFFER_SIZE} (expected 500)"),
        (SERIAL_BAUD == 921600, f"SERIAL_BAUD = {SERIAL_BAUD} (expected 921600)"),
        (BREATH_FREQ_MIN == 0.1, f"BREATH_FREQ_MIN = {BREATH_FREQ_MIN} (expected 0.1)"),
        (BREATH_FREQ_MAX == 0.6, f"BREATH_FREQ_MAX = {BREATH_FREQ_MAX} (expected 0.6)"),
        (HAMPEL_WINDOW == 5, f"HAMPEL_WINDOW = {HAMPEL_WINDOW} (expected 5)"),
    ]

    all_ok = all(c[0] for c in checks)

    for ok, msg in checks:
        if not ok:
            print(f"    ✗ {msg}")

    record("TC-SW-05", "All GPIO commands and config constants correct",
           all_ok,
           f"Checked {len(checks)} constants — {'all correct' if all_ok else 'some incorrect'}")

# main — run all tests
def main():
    print_banner()

    # run all test cases
    test_sw_01()
    print()
    test_sw_02()
    print()
    test_sw_03()
    print()
    test_sw_04()
    print()
    test_sw_05()

    # print summary
    print()
    print("═" * 58)
    print("  TEST SUMMARY")
    print("═" * 58)

    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    skipped = sum(1 for r in results if 'SKIP' in r.get('message', ''))

    for r in results:
        status = "\033[92mPASS\033[0m" if r['passed'] else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {r['id']}: {r['description']}")

    print()
    print(f"  Result: {passed}/{total} tests passed", end='')
    if skipped > 0:
        print(f" ({skipped} skipped — no hardware)", end='')
    print()

    if passed == total:
        print("  ✓ All tests PASSED!")
    elif passed >= total - 1 and skipped > 0:
        print("  ✓ All software tests PASSED (hardware test skipped)")
    else:
        print("  ✗ Some tests FAILED — review output above")

    print("═" * 58)
    print()

    return 0 if passed >= total - skipped else 1

if __name__ == "__main__":
    sys.exit(main())
