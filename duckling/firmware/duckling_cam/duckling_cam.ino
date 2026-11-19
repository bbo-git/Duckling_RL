// duckling_cam.ino - ESP32-CAM (AI-Thinker) van de Duckling. 2026-11-03.
//
// Ziet beweging (vision.h) en stuurt ~20x per seconde een CamPacket naar de RP2350 (duck_protocol.h).
// Krijgt van de RP2350 de kopstand en de draaiing van het lijf (om de eigen beweging eruit te halen).
//
// AFSTELLEN met je telefoon/laptop: wifi "Duckling-cam" (wachtwoord duckling1), dan http://192.168.4.1
//   Je ziet het camerabeeld (of het verschilbeeld), de gevonden vlek met kader, en alle getallen die naar
//   de RP2350 gaan. Met de schuifjes stel je drempels in; "bewaar" zet ze in het flashgeheugen.
//
// Board in de Arduino IDE: "AI Thinker ESP32-CAM" (esp32 core van Espressif). Flashen via een USB-serieel-
// adapter op U0T/U0R (IO0 naar GND tijdens het aanzetten). Niet IO13/IO14 gebruiken om te flashen.
// De code staat in duckling_cam_main.cpp (beeldverwerking: vision.h, protocol: duck_protocol.h).
