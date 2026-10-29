from machine import Pin, PWM
PINS = [2, 3, 4, 5, 6, 7]          # jouw servopinnen
servos = []
for p in PINS:
    s = PWM(Pin(p)); s.freq(50)
    s.duty_ns(1_500_000)           # 1500 us = midden
    servos.append(s)
print("alle servo's op 1500 us")