import numpy as np
from collections import deque

class AdaptiveDetector:
    
    def __init__(self, sensitivity=1.04, baseline_window=100, warmup_windows=20):
        self.sensitivity = sensitivity
        self.baseline_window = baseline_window
        self.warmup_windows = warmup_windows

        # sliding window of past measurements
        self.history = deque(maxlen=baseline_window)
        self.window_count = 0

        # track min values to estimate the "quiet" baseline
        self.quiet_history = deque(maxlen=baseline_window * 2)

    def detect_from_matrix(self, matrix):
        T, N = matrix.shape
        self.window_count += 1

        if T < 10:
            return self._empty_result("CALIBRATING...")

        # focus on the last 1 second for instant response
        # by only looking at the last 50 frames, a sudden movement isn't diluted
        # by the rest of the 10-second buffer. it spikes instantly.
        recent_frames = min(T, 50)
        recent_matrix = matrix[-recent_frames:]

        # metric 1: top-5 subcarrier temporal std
        subcarrier_stds = np.std(recent_matrix, axis=0)
        top5_std = float(np.mean(np.sort(subcarrier_stds)[-5:]))

        # metric 2: frame-to-frame difference energy
        diffs = np.diff(recent_matrix, axis=0)
        diff_energy = float(np.mean(np.abs(diffs)))

        # combined motion metric
        # weighted combination of both metrics
        motion_metric = top5_std * 0.7 + diff_energy * 100.0 * 0.3

        # update history
        self.history.append(motion_metric)
        self.quiet_history.append(motion_metric)

        # fft for display
        # use best subcarrier for fft visualization
        best_sc = np.argmax(subcarrier_stds)
        signal = matrix[:, best_sc].astype(np.float64)
        signal = signal - np.mean(signal)

        N_pad = max(2048, T * 4)
        spectrum = np.abs(np.fft.rfft(signal, n=N_pad))
        freqs = np.fft.rfftfreq(N_pad, d=1.0 / 50.0)

        peak_idx = np.argmax(spectrum[1:]) + 1  # skip dc
        peak_amplitude = float(spectrum[peak_idx])
        peak_freq = float(freqs[peak_idx])

        # warmup phase
        if self.window_count <= self.warmup_windows:
            return {
                'status': "ROOM EMPTY",
                'motion_metric': motion_metric,
                'baseline': motion_metric,
                'ratio': 1.0,
                'variance': motion_metric,
                'motion_energy': motion_metric,
                'peak_amplitude': peak_amplitude,
                'peak_freq': peak_freq,
                'confidence': 0,
                'freqs': freqs,
                'spectrum': spectrum,
                'is_warmup': True
            }

        # adaptive baseline
        # use the 25th percentile of history as the "quiet" baseline
        # this naturally tracks the lowest (most quiet) readings
        sorted_history = sorted(self.quiet_history)
        baseline = sorted_history[len(sorted_history) // 4]  # 25th percentile
        baseline = max(baseline, 0.001)  # prevent division by zero

        # detection: is current metric significantly above baseline?
        ratio = motion_metric / baseline

        if ratio > self.sensitivity:
            confidence = min(100, int((ratio - 1.0) / (self.sensitivity - 1.0) * 60))
            status = "HUMAN DETECTED"
        else:
            confidence = 0
            status = "ROOM EMPTY"

        return {
            'status': status,
            'motion_metric': motion_metric,
            'baseline': baseline,
            'ratio': ratio,
            'variance': motion_metric,
            'motion_energy': motion_metric,
            'peak_amplitude': peak_amplitude,
            'peak_freq': peak_freq,
            'confidence': confidence,
            'freqs': freqs,
            'spectrum': spectrum,
            'is_warmup': False
        }

    def _empty_result(self, status="ROOM EMPTY"):
        return {
            'status': status,
            'motion_metric': 0.0,
            'baseline': 0.0,
            'ratio': 0.0,
            'variance': 0.0,
            'motion_energy': 0.0,
            'peak_amplitude': 0.0,
            'peak_freq': 0.0,
            'confidence': 0,
            'freqs': np.array([]),
            'spectrum': np.array([]),
            'is_warmup': True
        }

# global detector instance (shared across modules)
_detector = AdaptiveDetector(sensitivity=1.04, baseline_window=100, warmup_windows=20)

def detect_from_matrix(matrix, threshold=None):
    return _detector.detect_from_matrix(matrix)

def detect(dominant_signal, sampling_rate=50, threshold=None):
    N = len(dominant_signal)
    if N < 4:
        return _detector._empty_result()

    # create a fake 2d matrix from the 1d signal
    matrix = dominant_signal.reshape(-1, 1)
    return _detector.detect_from_matrix(matrix)

def reset_detector(sensitivity=1.3, baseline_window=30, warmup_windows=5):
    global _detector
    _detector = AdaptiveDetector(sensitivity, baseline_window, warmup_windows)

if __name__ == "__main__":
    print("=== Adaptive Detector Self-Test ===\n")
    np.random.seed(42)

    detector = AdaptiveDetector(sensitivity=1.3, baseline_window=10, warmup_windows=3)

    # phase 1: warmup with quiet data
    print("Phase 1: Warmup (3 windows of quiet data)")
    for i in range(3):
        quiet = np.random.normal(50, 0.5, (500, 52)).astype(np.float32)
        result = detector.detect_from_matrix(quiet)
        print(f"  Window {i+1}: metric={result['motion_metric']:.4f} | warmup={result.get('is_warmup', False)}")

    # phase 2: more quiet data (should show empty)
    print("\nPhase 2: Quiet room (should be EMPTY)")
    for i in range(5):
        quiet = np.random.normal(50, 0.5, (500, 52)).astype(np.float32)
        result = detector.detect_from_matrix(quiet)
        print(f"  Window {i+4}: metric={result['motion_metric']:.4f} ratio={result['ratio']:.2f} | {result['status']}")

    # phase 3: human motion (should detect)
    print("\nPhase 3: Human motion (should DETECT)")
    t = np.linspace(0, 10, 500)
    for i in range(5):
        motion = np.random.normal(50, 0.5, (500, 52)).astype(np.float32)
        for sc in range(15):
            motion[:, sc] += np.sin(2 * np.pi * 0.5 * t + sc * 0.3) * 10
        result = detector.detect_from_matrix(motion)
        print(f"  Window {i+9}: metric={result['motion_metric']:.4f} ratio={result['ratio']:.2f} | {result['status']}")

    # phase 4: back to quiet
    print("\nPhase 4: Human leaves (should return to EMPTY)")
    for i in range(8):
        quiet = np.random.normal(50, 0.5, (500, 52)).astype(np.float32)
        result = detector.detect_from_matrix(quiet)
        print(f"  Window {i+14}: metric={result['motion_metric']:.4f} ratio={result['ratio']:.2f} | {result['status']}")

    print("\n=== Self-Test Complete ===")
