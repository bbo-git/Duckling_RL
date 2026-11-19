// duck_protocol.h - UART-protocol ESP32-CAM <-> RP2350 (2026-11-03). Zelfde bestand in beide sketches.
// 921600 baud, 8N1. ESP IO14 (TX) -> RP GPIO17 (RX), RP GPIO16 (TX) -> ESP IO13 (RX).
// Velden = die van de nep-ESP in de training (duckling_behavior.py, PersonSim.step).
#pragma once
#include <stdint.h>
#include <string.h>

#define CAM_H0 0xD7
#define CAM_H1 0x5A
#define RP_H0 0xA5
#define RP_H1 0x5A

#pragma pack(push, 1)
struct CamPacket {          // ESP -> RP, ~20 Hz
  uint8_t h0, h1, seq;
  uint8_t flags;            // bit0 = vis (er beweegt iets in beeld)
  int16_t bx_mrad;          // richting in het camerabeeld, + = links
  int16_t by_mrad;          // + = omhoog
  uint16_t dist_mm;         // afstand (grove schatting)
  uint8_t msize;            // bewegings-GROOTTE 0..3 (zoals de camera hem ziet)
  uint8_t mint;             // bewegings-INTENSITEIT 0..3
  int16_t tflow_mrad_s;     // eigen hoeksnelheid van het punt t.o.v. de wereld, + = naar links
  int16_t approach_mm_s;    // naderingssnelheid, + = komt dichterbij
  uint8_t fps;              // beelden per seconde (controle)
  uint8_t sum;              // som van alle voorgaande bytes (mod 256)
};
struct RpPacket {           // RP -> ESP, 50 Hz
  uint8_t h0, h1, seq;
  uint8_t flags;            // bit0 = fixeer (kop staat stil), bit1 = wifi van de cam AAN
  int16_t head_yaw_mrad;    // kop-yaw t.o.v. het lijf
  int16_t head_elev_mrad;   // kop-elevatie (nek + kop-pitch), + = omhoog
  int16_t head_yaw_rate_mrad_s;
  int16_t gyro_z_mrad_s;    // draaien van het lijf (IMU), + = linksom
  uint8_t lamp;             // flitsled van de cam, 0..100 % (2026-11-03)
  uint8_t sum;
};
#pragma pack(pop)

static inline uint8_t duck_sum(const void* p, int n) {
  const uint8_t* b = (const uint8_t*)p; uint8_t s = 0;
  for (int i = 0; i < n; i++) s += b[i];
  return s;
}

// Eenvoudige ontvanger: stop bytes erin, true zodra er een geldig pakket in 'out' staat.
template <typename P, uint8_t H0, uint8_t H1>
struct PacketReader {
  uint8_t buf[sizeof(P)]; int n = 0;
  uint32_t ok = 0, bad = 0;
  bool feed(uint8_t c, P& out) {
    if (n == 0 && c != H0) return false;
    if (n == 1 && c != H1) { n = (c == H0) ? 1 : 0; return false; }
    buf[n++] = c;
    if (n < (int)sizeof(P)) return false;
    n = 0;
    if (duck_sum(buf, sizeof(P) - 1) != buf[sizeof(P) - 1]) { bad++; return false; }
    memcpy(&out, buf, sizeof(P)); ok++;
    return true;
  }
};
