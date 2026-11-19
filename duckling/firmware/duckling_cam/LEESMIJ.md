# Duckling-cam (ESP32-CAM, AI-Thinker)

Ziet beweging en stuurt ~20x per seconde naar de RP2350: zichtbaar, richting (bx, by), afstand,
grootte 0–3, intensiteit 0–3, eigen beweging en nadering (`duck_protocol.h`, dezelfde velden als in de training).

## Installeren (eenmalig)
1. Arduino IDE → Voorkeuren → extra boardbeheer-URL:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
2. Boardbeheer: **esp32** (Espressif) installeren. Getest (compileren) met 3.0.7: 1,0 MB flash, 94 kB RAM.
3. Board: **AI Thinker ESP32-CAM**, en onder Extra → **Partition Scheme: "Minimal SPIFFS (1.9MB APP with OTA)"**
   (nodig voor draadloos updaten; de standaard "Huge APP" kan dat niet).

## Flashen
USB-serieeladapter (3,3 V!): adapter-TX → **U0R**, adapter-RX → **U0T**, GND → GND, 3,3 V → 3V3.
**IO0 aan GND** tijdens het aanzetten = flash-modus. Uploaden, dan IO0 los en opnieuw aanzetten.
(IO13/IO14 blijven aan de RP2350; die worden niet gebruikt om te flashen.)

## Bijwerken via de browser (aanbevolen, geen firewall-gedoe)
1. Arduino IDE: Schets → **Gecompileerd binair bestand exporteren** (Partition Scheme "Minimal SPIFFS ... with OTA").
   Het bestand komt in `duckling_cam/build/esp32.esp32.esp32cam/duckling_cam.ino.bin`.
2. Wifi **Duckling-cam**, browser **http://192.168.4.1/update** (of de link op de afstelpagina),
   inloggen: gebruiker **duck**, wachtwoord **duckling1**.
3. Kies `duckling_cam.ino.bin` (NIET de .bootloader.bin of .partitions.bin) → Update. Hij start vanzelf opnieuw.

## Draadloos updaten vanuit de Arduino IDE (OTA) - na de eerste keer met de FT232
(Lukt dit niet met "No response from device": dan blokkeert de Windows-firewall de terugverbinding - gebruik de browser.)
1. Laptop met wifi **Duckling-cam** verbinden.
2. Arduino IDE → Extra → Poort → netwerkpoort **duckling-cam at 192.168.4.1** (kan even duren voor hij verschijnt).
3. Upload; wachtwoord **duckling1**. Partition Scheme moet "Minimal SPIFFS ... with OTA" blijven.
De wifi moet aan staan: zonder RP2350 altijd, met RP2350: `wifi aan` in de seriële monitor van de RP2350.

## Flitsled
De RP2350 stuurt de helderheid (commando `lamp <0-100>`; later het gedrag). Na elke verandering kijkt de
cam ~0,25 s niet (de belichting past zich aan), anders lijkt het hele beeld te bewegen.

## Afstellen (zonder RP2350 kan ook)
Telefoon/laptop: wifi **Duckling-cam**, wachtwoord **duckling1**, browser: **http://192.168.4.1**
- beeld / verschilbeeld, rood kader = wat hij als "beweging" ziet, groene stip = het midden;
- alle getallen die naar de RP2350 gaan;
- schuifjes: `diff_thr` (gevoeligheid), `min_energy` (vanaf wanneer hij iets ziet),
  `i1/i2/i3` (intensiteit 1/2/3), `s1/s2/s3` (grootte 1/2/3 als fractie van het beeld),
  `rot` (0/90/180/270: camera gedraaid gemonteerd - zet hem zo dat het beeld rechtop staat),
  `flip_x/flip_y` (gespiegeld), `lag` (1–3: hoger = langzame beweging beter zichtbaar).
  Let op: bij 90/270 ziet hij opzij minder ver (±22 gr i.p.v. ±30 gr) - het beeld is dan staand.
  **bewaar** zet ze in het flashgeheugen.

Doel bij het afstellen (= wat de eend in de training leerde), op ongeveer 50 cm:
| wat je doet | grootte | intensiteit |
|---|---|---|
| niets | – (niet zichtbaar) | – |
| praten, klein gebaar | 1 | 1 |
| hand zwaaien | 1 | 2 |
| hard zwaaien | 1 | 3 |
| lopen / dansen (heel lijf) | 3 | 3 |

Richting: iets **links** in beeld (gezien vanuit de eend) moet een **positieve** richting geven. Anders `flip_x`.

Wifi uit (scheelt ~100 mA): `WIFI_AAN false` bovenin `duckling_cam_main.cpp`.
Test op de pc: `tools/test_vision.cpp` (nagemaakte beelden: stil, kop draait, hand links/rechts, lopen, naderen).
