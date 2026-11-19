# Duckling loop-firmware (RP2350, Arduino)

Netwerk: zie `policy_weights.h` (nu `duckling_walk_1600.onnx`). Ijken blijft `ijk.py` (MicroPython).

## Installeren (eenmalig)
1. Arduino IDE 2 → Bestand → Voorkeuren → extra boardbeheer-URL:
   `https://github.com/earlephilhower/arduino-pico/releases/download/global/package_rp2040_index.json`
2. Boardbeheer: **Raspberry Pi Pico/RP2040/RP2350** (Earle Philhower) installeren.
3. Board: **Waveshare RP2350 Core** (RP2350B, 16 MB). Getest met core 6.3.2: compileert, 315 kB flash, 331 kB RAM.

## Uploaden
Open `duckling_loop.ino`, Upload (BOOT ingedrukt houden bij het insteken als hij niet gevonden wordt).
Terug naar MicroPython voor `ijk.py`: MicroPython-.uf2 er weer op zetten. calib.json op het bordje blijft niet
bewaard bij het wisselen: bewaar hem eerst op de pc (`toon` in ijk.py).

## Na een nieuwe ijking of een ander netwerk
```
python tools/maak_firmware_data.py --onnx ../onnx/duckling_walk_1600.onnx ^
  --grid src/mjlab_microduck/robot/duckling/grids/gait_grid_normaal_v3.npz ^
  --calib firmware/calib.json --uit firmware/duckling_loop
```
en opnieuw uploaden. Controle op de pc (optioneel): `tools/test_core.cpp` + `tools/test_core.py`
(netwerk in C tegen onnxruntime: verschil 2e-6 rad; servo-omrekening tegen ijk.py: 0.001 µs).

## Eerste keer aanzetten (Serial Monitor, 115200, Newline)
1. **Accu UIT, USB erin** → upload → `status`. Servo's staan SLAP.
2. `imu ijk`: servo's gaan naar HOME, zet hem rechtop op tafel, 1x Enter. Bepaalt of IMU-z omhoog/omlaag wijst, het gyro-nulpunt en de
   scheefstand (IMU-x = neus). Controle met `imu`: rechtop zwaartekracht ≈ (0, 0, −1); voorover x > 0; linkerkant omlaag y > 0;
   linksom draaien gyro z > 0.
3. **Voeding tijdens het testen (USB + accu tegelijk):** de accu mag NIET aan VB als USB erin zit (geen diode).
   Daarom: accu alleen naar de **servo's**, de draad accu → VB losmaken (of de P-MOSFET uit OVERDRACHT.md
   plaatsen), GND gemeenschappelijk laten. Het bordje draait dan op USB, de servo's op de accu.
   Hang hem aan een touwtje, voeten los.
4. `thuis` → staan alle gewrichten in HOME? Dan `anim 0.3` → `anim 1`: loopt het loopje goed, links/rechts gespiegeld?
   Eén gewricht verkeerd om = in `ijk.py` met `draai` omdraaien.
5. Voeten net op tafel: `staan`. Rustig, geen zoemen? Dan touw iets vieren: `loop 0.3`, `draai 20`, `draai -20`.
6. Gevallen? Hij gaat zelf slap. `log` → kopieer de CSV en stuur hem op.

Knopje GPIO21: kort = noodstop (slap), 2 s vasthouden terwijl hij slap is = staan.
