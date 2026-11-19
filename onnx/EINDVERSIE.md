# Duckling V2 - eindversie loopnetwerk (2026-11-01)

Code: duck_v2_compleet30.zip (training + gedrag + player). Run 2026-11-01_08-18-31, hervat vanaf 950 met compleet29/30.

| checkpoint | goede draairichting | draaisnelheid (links/rechts) | tenen (heup-yaw) | bal aankijken (50 gr opzij) | valt |
|---|---|---|---|---|---|
| 1150 | 22/24 | 31 % / 43 % | -2 gr | 4/4, na 3.6-5 s | nee |
| 1600 | 12/24 | ~0 % / 16 % | -8 gr | 0/4 | nee |

Gemeten met tools/draaitest.py, evalpol2/evalrew (trainingsomgeving) en face-test (player-fysica + gedrag).
Keuze (Jairo, 2026-11-10): **1150** - zag er in de player iets stabieler uit en draait beter. (Eerder 1600 gekozen.)
Firmware: firmware/duckling_loop is gemaakt met 1150 (duckling_loop.ino.uf2).
