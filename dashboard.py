import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # ensure interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

from config import (
    BUFFER_SIZE,
    SAMPLING_RATE,
    DETECTION_THRESHOLD
)
from hampel_filter import hampel_filter_vectorized
from preprocessor import preprocess
from detector import detect_from_matrix, detect

def create_dashboard(receiver, detector_fn=None, gpio_ctrl=None, threshold=None):
    if detector_fn is None:
        detector_fn = detect
    if threshold is None:
        threshold = DETECTION_THRESHOLD

    # apply dark theme
    plt.style.use('dark_background')

    # create figure with custom grid
    fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')
    fig.canvas.manager.set_window_title('WiFi Radar Dashboard — XIAO ESP32-C6')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.96, top=0.93, bottom=0.06)

    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_fft = fig.add_subplot(gs[0, 1])
    ax_status = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])

    # style helper
    title_color = '#58a6ff'
    text_color = '#c9d1d9'
    green_color = '#3fb950'
    red_color = '#f85149'
    amber_color = '#d29922'
    bg_color = '#0d1117'
    panel_color = '#161b22'

    for ax in [ax_heatmap, ax_fft, ax_status, ax_metrics]:
        ax.set_facecolor(panel_color)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # initialize plots
    # heatmap placeholder
    blank_matrix = np.zeros((BUFFER_SIZE, 52))
    img = ax_heatmap.imshow(blank_matrix, aspect='auto', cmap='hot',
                            interpolation='nearest', vmin=0, vmax=80)
    ax_heatmap.set_title('CSI Amplitude — Last 10 seconds',
                         color=title_color, fontsize=11, fontweight='bold', pad=8)
    ax_heatmap.set_xlabel('Subcarrier Index', color=text_color, fontsize=9)
    ax_heatmap.set_ylabel('Time Sample', color=text_color, fontsize=9)
    cbar = fig.colorbar(img, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=text_color, labelsize=8)
    cbar.set_label('Amplitude', color=text_color, fontsize=9)

    # frequency plot (full spectrum, no breathing band)
    fft_line, = ax_fft.plot([], [], color='#58a6ff', linewidth=1.5, alpha=0.9)
    peak_line = ax_fft.axvline(x=0, color=red_color, linestyle='--',
                               linewidth=1.5, alpha=0.8, label='Peak')
    ax_fft.set_xlim(0, 5.0)
    ax_fft.set_title('Signal Frequency Analysis (Motion Energy)',
                     color=title_color, fontsize=11, fontweight='bold', pad=8)
    ax_fft.set_xlabel('Frequency (Hz)', color=text_color, fontsize=9)
    ax_fft.set_ylabel('Energy Magnitude', color=text_color, fontsize=9)
    ax_fft.legend(loc='upper right', fontsize=8, facecolor=panel_color,
                  edgecolor='#30363d', labelcolor=text_color)

    # status display
    ax_status.set_xlim(0, 10)
    ax_status.set_ylim(0, 10)
    ax_status.axis('off')
    status_text = ax_status.text(5, 7.5, 'INITIALIZING...', fontsize=20,
                                 fontweight='bold', ha='center', va='center',
                                 color=amber_color,
                                 fontfamily='monospace')
    activity_text = ax_status.text(5, 5.5, '', fontsize=14,
                               ha='center', va='center', color=text_color,
                               fontfamily='monospace')
    # confidence bar background
    conf_bg = ax_status.barh(2.5, 10, height=1.2, color='#21262d',
                             edgecolor='#30363d', linewidth=1)
    conf_bar = ax_status.barh(2.5, 0, height=1.2, color=green_color, alpha=0.8)
    conf_label = ax_status.text(5, 1.0, 'Presence Confidence: 0%', fontsize=10,
                                ha='center', va='center', color=text_color,
                                fontfamily='monospace')
    ax_status.set_title('Presence Status',
                        color=title_color, fontsize=11, fontweight='bold', pad=8)

    # metrics display
    ax_metrics.set_xlim(0, 10)
    ax_metrics.set_ylim(0, 10)
    ax_metrics.axis('off')
    ax_metrics.set_title('System Metrics',
                         color=title_color, fontsize=11, fontweight='bold', pad=8)

    pca_label = ax_metrics.text(5, 8.0, 'PCA Explained Variance', fontsize=10,
                                ha='center', va='center', color=text_color,
                                fontfamily='monospace')
    pca_bar_bg = ax_metrics.barh(6.5, 10, height=1.0, color='#21262d',
                                 edgecolor='#30363d', linewidth=1)
    pca_bar = ax_metrics.barh(6.5, 0, height=1.0, color='#58a6ff', alpha=0.8)
    pca_value = ax_metrics.text(5, 5.2, '0.0%', fontsize=12,
                                ha='center', va='center', color=text_color,
                                fontweight='bold', fontfamily='monospace')

    rate_text = ax_metrics.text(5, 3.5, 'Packet Rate: -- pkt/s', fontsize=11,
                                ha='center', va='center', color=text_color,
                                fontfamily='monospace')
    buffer_text = ax_metrics.text(5, 2.2, 'Buffer: 0/500', fontsize=10,
                                  ha='center', va='center', color=text_color,
                                  fontfamily='monospace')
    outlier_text = ax_metrics.text(5, 1.0, 'Variance: 0 | Threshold: 0', fontsize=10,
                                   ha='center', va='center', color='#d29922',
                                   fontfamily='monospace', fontweight='bold')

    # suptitle
    fig.suptitle('WiFi Radar — Through-Wall Human Detection',
                 fontsize=14, fontweight='bold', color=title_color, y=0.97)

    # state tracking
    state = {
        'consecutive_detections': 0,
        'consecutive_empty': 0,
        'is_present': False,
        'total_windows': 0,
        'detection_count': 0,
        'empty_count': 0
    }

    # animation update function
    def update(frame):
        matrix = receiver.get_matrix()

        if matrix is None:
            # buffer not yet full — show filling status
            fill = receiver.buffer_fill
            status_text.set_text(f'BUFFERING... {fill}/{BUFFER_SIZE}')
            status_text.set_color(amber_color)
            activity_text.set_text('Scanning environment...')
            buffer_text.set_text(f'Buffer: {fill}/{BUFFER_SIZE}')
            rate_text.set_text(f'Packet Rate: {receiver.packet_rate} pkt/s')
            return [img, fft_line, peak_line, status_text, activity_text,
                    conf_label, pca_value, rate_text, buffer_text, outlier_text]

        state['total_windows'] += 1

        # stage 2: hampel filter
        cleaned, outlier_count = hampel_filter_vectorized(matrix)

        # stage 3: pca preprocessing
        dominant_signal, explained_var = preprocess(cleaned)

        # stage 4: adaptive self-calibrating detection
        result = detect_from_matrix(cleaned)

        # hysteresis logic for stable status
        if result.get('is_warmup', False):
            # during warmup, show calibrating
            status_text.set_text('AUTO-CALIBRATING...')
            status_text.set_color(amber_color)
            activity_text.set_text(f'Learning baseline... {state["total_windows"]}/20')
            activity_text.set_color(amber_color)
        elif result['status'] == "HUMAN DETECTED":
            state['consecutive_detections'] += 1
            state['consecutive_empty'] = 0
            if state['consecutive_detections'] >= 1:  # instant reaction
                state['is_present'] = True
            state['detection_count'] += 1
        else:
            state['consecutive_empty'] += 1
            state['consecutive_detections'] = 0
            if state['consecutive_empty'] >= 1:       # instant reaction
                state['is_present'] = False
            state['empty_count'] += 1

        # debug: print adaptive metrics to console
        ratio = result.get('ratio', 0)
        baseline = result.get('baseline', 0)
        metric = result.get('motion_metric', 0)
        status_str = 'WARMUP' if result.get('is_warmup') else ('PRESENT' if state['is_present'] else 'EMPTY')
        print(f"  Metric={metric:7.2f} | Base={baseline:7.2f} | Ratio={ratio:5.2f} | {status_str}")

        # stage 6: gpio control
        if gpio_ctrl is not None and not result.get('is_warmup', False):
            gpio_ctrl.handle_result(result, state['consecutive_detections'])

        # update heatmap
        img.set_data(cleaned)
        img.set_clim(vmin=np.percentile(cleaned, 5),
                     vmax=np.percentile(cleaned, 95))

        # update fft plot
        if len(result['freqs']) > 0 and len(result['spectrum']) > 0:
            fft_line.set_data(result['freqs'], result['spectrum'])
            ax_fft.set_xlim(0, 5.0)
            max_mag = max(np.max(result['spectrum']) * 1.2, 1.0)
            ax_fft.set_ylim(0, max_mag)

            if result['peak_freq'] > 0:
                peak_line.set_xdata([result['peak_freq']])
                peak_line.set_alpha(0.8)
            else:
                peak_line.set_alpha(0.0)

        # update status panel (using hysteresis state)
        if not result.get('is_warmup', False):
            if state['is_present']:
                status_text.set_text('● HUMAN PRESENT')
                status_text.set_color(green_color)

                var = result.get('ratio', 1.0)
                level = "HIGH" if var > 2.0 else ("MEDIUM" if var > 1.5 else "LOW")
                activity_text.set_text(f'Activity Level: {level}')
                activity_text.set_color(green_color)

                conf_pct = result['confidence'] / 100.0 * 10
                conf_bar[0].set_width(conf_pct)
                conf_bar[0].set_color(green_color)
                conf_label.set_text(f'Presence Confidence: {result["confidence"]}%')
            else:
                status_text.set_text('○ NO PRESENCE')
                status_text.set_color(red_color)
                activity_text.set_text('Monitoring environment...')
                activity_text.set_color('#8b949e')
                conf_bar[0].set_width(0)
                conf_label.set_text('Presence Confidence: 0%')

        # update metrics panel
        pca_width = min(explained_var / 100.0 * 10, 10)
        pca_bar[0].set_width(pca_width)
        pca_value.set_text(f'{explained_var:.1f}%')

        rate_text.set_text(f'Packet Rate: {receiver.packet_rate} pkt/s')
        buffer_text.set_text(f'Buffer: {receiver.buffer_fill}/{BUFFER_SIZE}')
        outlier_text.set_text(f'Ratio: {ratio:.2f} | Baseline: {baseline:.1f}')

        return [img, fft_line, peak_line, status_text, activity_text,
                conf_label, pca_value, rate_text, buffer_text, outlier_text]

    # create animation — update every 250ms for high responsiveness
    ani = FuncAnimation(fig, update, interval=250, blit=False,
                        cache_frame_data=False)

    return fig, ani, state

if __name__ == "__main__":
    # demo mode with synthetic data (no hardware needed)
    print("=== Dashboard Demo Mode (synthetic data) ===")
    print("Close the window to exit.\n")

    import time
    from collections import deque

    class MockReceiver:
        def __init__(self):
            self._t = 0
            self._packet_timestamps = deque(maxlen=200)

        def get_matrix(self):
            t_axis = np.linspace(self._t, self._t + 10, BUFFER_SIZE)
            self._t += 1
            matrix = np.random.normal(50, 3, (BUFFER_SIZE, 52)).astype(np.float32)
            # add motion signal to first 10 subcarriers
            for sc in range(10):
                matrix[:, sc] += np.sin(2 * np.pi * 0.5 * t_axis) * (15 - sc)
            return matrix

        @property
        def packet_rate(self):
            return 50

        @property
        def buffer_fill(self):
            return BUFFER_SIZE

        def send_gpio(self, cmd):
            pass

    mock = MockReceiver()
    fig, ani, state = create_dashboard(mock, threshold=5.0)
    plt.show()
