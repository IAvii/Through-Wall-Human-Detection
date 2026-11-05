import serial, time

port = "COM7"
ser = serial.Serial(port, 921600, timeout=1)

print("Waiting 10 seconds for ESP32 to reboot and connect to WiFi...")
time.sleep(10)

print("GREEN LED ON...")
ser.write(b'G')
time.sleep(3)

print("RED LED ON...")
ser.write(b'R')
time.sleep(3)

print("ALL OFF")
ser.write(b'X')

ser.close()
print("Done!")
