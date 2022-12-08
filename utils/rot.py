import serial.tools.list_ports
import time


ports = list(serial.tools.list_ports.comports())
port = None
for p in ports:
    p = str(p)
    if "Uno" in p:
        port = p.split(' ')[0]

print('found uno at: ', port)

uno = serial.Serial(
    port=port,
    baudrate=9600,
    timeout=1,
)

uno.flushInput()

while True:
    if uno.inWaiting():
        b = uno.read(uno.inWaiting()).decode("gbk").split()[0]
        b = float(b)
        if b > 1 or b < 0:
            continue
        print(b)
    time.sleep(0.1)
