from config import (
    GPIO_GREEN,
    GPIO_RED,
    GPIO_BUZZ,
    GPIO_RESET,
    MIN_CONSECUTIVE_DETECTIONS
)

class GPIOController:
    
    def __init__(self, receiver, enabled=True):
        self.receiver = receiver
        self.enabled = enabled
        self.last_command = ''
        self.command_count = 0

    def _send(self, cmd):
        self.last_command = cmd
        if self.enabled:
            self.receiver.send_gpio(cmd)
            self.command_count += 1

    def green_on(self):
        self._send(GPIO_GREEN)

    def red_on(self):
        self._send(GPIO_RED)

    def buzz(self):
        self._send(GPIO_BUZZ)

    def reset(self):
        self._send(GPIO_RESET)

    def handle_result(self, result, consecutive_count):
        if result['status'] == "HUMAN DETECTED":
            self.green_on()
            if consecutive_count >= MIN_CONSECUTIVE_DETECTIONS:
                self.buzz()
        else:
            self.red_on()

if __name__ == "__main__":
    # demonstrate gpio controller without hardware
    print("=== GPIO Controller Demo (no hardware) ===\n")

    class MockReceiver:
        def send_gpio(self, cmd):
            cmd_names = {'G': 'GREEN LEDs ON', 'R': 'RED LEDs ON',
                         'B': 'BUZZER PULSE', 'X': 'ALL OFF'}
            print(f"  → Serial TX: '{cmd}' = {cmd_names.get(cmd, 'UNKNOWN')}")

    mock = MockReceiver()
    gpio = GPIOController(mock, enabled=True)

    # simulate detection sequence
    print("Simulating 5-window detection sequence:\n")

    results = [
        {'status': 'ROOM EMPTY'},
        {'status': 'HUMAN DETECTED'},
        {'status': 'HUMAN DETECTED'},
        {'status': 'HUMAN DETECTED'},  # 3rd consecutive → buzzer
        {'status': 'ROOM EMPTY'},
    ]

    consecutive = 0
    for i, result in enumerate(results):
        if result['status'] == 'HUMAN DETECTED':
            consecutive += 1
        else:
            consecutive = 0

        print(f"Window {i+1}: {result['status']} (consecutive: {consecutive})")
        gpio.handle_result(result, consecutive)
        print()

    print(f"Total GPIO commands sent: {gpio.command_count}")
    print("\n=== Demo Complete ===")
