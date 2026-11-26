// duckling_loop.ino - loop-firmware van de Duckling V2 (RP2350, Arduino-pico core). 2026-11-02.
//
// Wat het doet: IMU (GY-85: ITG-3205 gyro + ADXL345) 200 Hz -> zwaartekrachtschatting; elke 20 ms het
// loopnetwerk (exact als de training, zie duck_core.h) -> servo's. Zonder camera: opdrachten via USB-serieel.
//
// COMMANDO'S (Serial Monitor, 115200, "Newline"):
//   slap            alle servo's slap (ook: knopje GPIO21 kort indrukken = NOODSTOP)
//   thuis           alle servo's naar HOME (de CAD-nulstand) en vasthouden
//   zet <g> <gr>    een gewricht (bv. left_hip_roll) langzaam naar <gr> t.o.v. HOME: tekencontrole
//   wip [gr]        alle gewrichten tegelijk HOME <-> +gr (standaard 10), elke 1,5 s; 'thuis' = stop
//   anim [a]        loopje in open lus afspelen (voor de test HANGEND, voeten los), a = amplitude 0..1
//   staan / opstaan netwerk aan, stilstaan (rust: HOME vast; vanuit zitten: opstaan)
//   zit             gaan zitten (vaste beweging naar zit_v2), 'staan' = opstaan
//   rust aan|uit    aan = stilstaan zonder netwerk (= player), uit = netwerk ook bij stilstaan (balanceren)
//   loop <f> [w] [s] netwerk, vooruit met f x 100%-snelheid (bv. loop 0.3), w = draaien in gr/s (+ = links),
//                   s = duur in seconden, daarna vanzelf staan (bv. loop 0.3 0 10)
//   draai <w> [s]   op de plaats draaien, w in gr/s, s = duur in seconden
//   passen <n> [f]  n passen vooruit (standaard op 30 %), dan vanzelf staan (bv. passen 3)
//   oefen [k] [n] [f] [rust]  k keer: n passen op f, dan rust (standaard 5 1 0.25 1.5) = pasje, balans, pasje...
//   imu             5 s lang gyro en zwaartekracht in romp-assen tonen (controle van de richtingen)
//   imu ijk         IMU ijken: rechtop op tafel in HOME, 1x Enter (richting z, nulpunt, scheefstand), wordt bewaard
//   imu test        controle gyro-richting t.o.v. versnellingsmeter (kantelen + draaien), fout = omdraaien
//   imu nul         gyro-nulpunt opnieuw meten (robot stil laten staan)
//   log             laatste ~20 s als CSV (na een val: de seconden ervoor), log wis = opnieuw beginnen
//   status          stand van zaken
//   cam             3 s lang wat de camera (ESP32-CAM) stuurt
//   wifi aan|uit    wifi van de camera (afstelpagina) aan of uit; wordt bewaard
//   lamp <0-100>    flitsled van de camera (helderheid in %)
// Knopje GPIO21: kort = slap (noodstop). Lang (2 s) terwijl hij slap is = staan.
// Valbeveiliging: romp > 35 gr gekanteld -> slap, log bevroren. Netwerk start niet bij > 15 gr.  Accu < 3.45 V -> slap.
#include <Arduino.h>
#include <Wire.h>
#include <EEPROM.h>
#include <hardware/pwm.h>
#include <hardware/clocks.h>
#include "duck_core.h"
#include "duck_protocol.h"

// ------------------------------------------------------------------ instellingen
#define PIN_BTN 21
#define PIN_BAT 40          // ADC, deler 100k/100k
#define PIN_SDA 18
#define PIN_SCL 19
#define IMU_HZ 200
#define CTRL_EVERY (IMU_HZ / 50)
#define FALL_DEG 35.0f
#define START_MAX_DEG 15.0f  // netwerk start alleen als hij (volgens de IMU) rechtop staat
#define BAT_MIN 3.45f
#define ACC_V 0.15f         // m/s2  opdracht verandert geleidelijk (= gedrag)
#define ACC_W 1.5f          // rad/s2
#define ENABLE_STAGGER_MS 50
#define GRAV_K 0.02f        // complementair filter: aandeel versnellingsmeter per IMU-stap
#define PIN_CAM_TX 16       // -> ESP IO13
#define PIN_CAM_RX 17       // <- ESP IO14
#define CAM_BAUD 921600

enum Mode { SLAP, THUIS, ANIM, NET };
static const char* MODE_NAAM[] = {"SLAP", "THUIS", "ANIM", "NETWERK"};

static Policy pol;
static Duck duck;
static Mode mode = SLAP;
static uint32_t mode_t0 = 0;
static float tw_goal[3] = {0, 0, 0}, tw[3] = {0, 0, 0};
static uint32_t tw_stop_ms = 0;
static int steps_left = 0; static float step_ph = 0;
static bool step_wait = false; static int last_half = -1; static float step_v = 0;   // 2026-11-26: = training: start op een halve cyclus, om en om
static int oefen_left = 0, oefen_n = 1; static float oefen_f = 0.25f, oefen_rust = 1.5f; static uint32_t oefen_t = 0;   // 2026-11-26: oefen   // 2026-11-26: passen <n>: na n passen (halve gangcycli) staan   // 2026-11-26: loop/draai met duur: daarna vanzelf opdracht 0 (staan)
// 2026-11-18: zoals tools/duckling_gedrag.py: stilstaan = HOME vasthouden (rust), netwerk alleen bij lopen
// (in 0,25 s erin, in 1 s eruit); zitten/opstaan = vaste beweging (min-jerk) naar de zithouding zit_v2.
enum Post { STAAT, GAAT_ZITTEN, ZIT, STAAT_OP };
static const char* POST_NAAM[] = {"STAAT", "GAAT ZITTEN", "ZIT", "STAAT OP"};
static Post post = STAAT; static float post_t = 0, sit_amt = 0, rest_b = 0;
static bool want_sit = false, rust_aan = false;   // 2026-11-26 (Jairo): netwerk ook aan in rust (balanceren); "rust aan" = oude rust-modus
#define SIT_S 0.5f
#define SIT_UP_S 0.833f
#define FADE_IN_S 0.25f
#define FADE_OUT_S 1.0f
static const float CROUCH[14] = {   // zit_v2 (sit_gen.py), rad t.o.v. HOME: L yaw roll pitch knie enkel, nek, kop-pitch, kop-roll, kop-yaw, R ...
  0.0f, 0.0f, 0.42097f, -0.80896f, -0.32865f, -0.50004f, -0.49393f, -0.02443f, 0.0f,
  0.0f, 0.0f, 0.42097f, -0.80896f, -0.32865f};
static inline float min_jerk(float u) { u = u < 0 ? 0 : (u > 1 ? 1 : u); return u * u * u * (10 - 15 * u + 6 * u * u); }
static float head_cmd[4] = {0, 0, 0, 0};
static float anim_amp = 1.0f, anim_a = 0.0f, anim_ph = 0.0f;
static float q_out[NJ];                 // gewrichtsdoelen die naar de servo's gaan
static float q_zet[NJ] = {0};           // 2026-11-18: doelstand in THUIS (commando zet), 0 = HOME
static float wip_deg = 0; static uint32_t wip_t0 = 0;   // commando wip: alle gewrichten 0 <-> +wip_deg
static bool servo_on[N_SERVO];
static float t_infer_us = 0, t_loop_max_us = 0;
static float bat_v = 0; static uint32_t bat_low_t = 0;


// ------------------------------------------------------------------ IMU (GY-85)
#define ITG 0x68
#define ADXL 0x53
static int8_t M_ACC[9], M_GYR[9];       // romp = M * sensor (rijen: romp x, y, z)
static float LEVEL[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // 2026-11-18: kleine scheefstand van de IMU wegwerken (rechtop = 0,0,-1)
static float gyro_bias[3] = {0, 0, 0};
static float gyro_b[3], acc_b[3], grav[3] = {0, 0, -1};

static void wr(uint8_t a, uint8_t r, uint8_t v) { Wire1.beginTransmission(a); Wire1.write(r); Wire1.write(v); Wire1.endTransmission(); }
static bool rd(uint8_t a, uint8_t r, uint8_t* b, int n) {
  Wire1.beginTransmission(a); Wire1.write(r);
  if (Wire1.endTransmission(false) != 0) return false;
  if (Wire1.requestFrom(a, (uint8_t)n) != n) return false;
  for (int i = 0; i < n; i++) b[i] = Wire1.read();
  return true;
}
static void imu_init() {
  Wire1.setSDA(PIN_SDA); Wire1.setSCL(PIN_SCL); Wire1.begin(); Wire1.setClock(400000);
  wr(ITG, 0x3E, 0x01);          // klok: PLL gyro X
  wr(ITG, 0x15, 4);             // 1 kHz / 5 = 200 Hz
  wr(ITG, 0x16, 0x18 | 0x03);   // +-2000 gr/s, laagdoorlaat 42 Hz
  wr(ADXL, 0x2C, 0x0B);         // 200 Hz
  wr(ADXL, 0x31, 0x0B);         // full resolution, +-16 g (256 LSB/g)
  wr(ADXL, 0x2D, 0x08);         // meten
}
static bool imu_raw(float g[3], float a[3]) {   // sensor-assen: gyro rad/s, acc in g
  uint8_t b[6];
  if (!rd(ITG, 0x1D, b, 6)) return false;
  for (int i = 0; i < 3; i++) g[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]) / 14.375f * 0.01745329f;
  if (!rd(ADXL, 0x32, b, 6)) return false;
  for (int i = 0; i < 3; i++) a[i] = (int16_t)(b[2 * i] | (b[2 * i + 1] << 8)) / 256.0f;
  return true;
}
static void level3(float* v) {
  float o[3];
  for (int r = 0; r < 3; r++) o[r] = LEVEL[3 * r] * v[0] + LEVEL[3 * r + 1] * v[1] + LEVEL[3 * r + 2] * v[2];
  v[0] = o[0]; v[1] = o[1]; v[2] = o[2];
}
static void map3(const int8_t* M, const float* s, float* o) {
  for (int r = 0; r < 3; r++) o[r] = M[3 * r] * s[0] + M[3 * r + 1] * s[1] + M[3 * r + 2] * s[2];
}
static bool imu_update(float dt) {
  float gs[3], as[3];
  if (!imu_raw(gs, as)) return false;
  for (int i = 0; i < 3; i++) gs[i] -= gyro_bias[i];
  map3(M_GYR, gs, gyro_b); map3(M_ACC, as, acc_b); level3(gyro_b); level3(acc_b);
  // zwaartekracht (romp-assen): d g / dt = -w x g, bijgestuurd met -acc/|acc|
  float w0 = gyro_b[0], w1 = gyro_b[1], w2 = gyro_b[2];
  float c0 = w1 * grav[2] - w2 * grav[1], c1 = w2 * grav[0] - w0 * grav[2], c2 = w0 * grav[1] - w1 * grav[0];
  grav[0] -= c0 * dt; grav[1] -= c1 * dt; grav[2] -= c2 * dt;
  float an = sqrtf(acc_b[0] * acc_b[0] + acc_b[1] * acc_b[1] + acc_b[2] * acc_b[2]);
  if (an > 0.5f && an < 1.5f)   // alleen als hij niet hard schokt
    for (int i = 0; i < 3; i++) grav[i] += GRAV_K * (-acc_b[i] / an - grav[i]);
  float n = sqrtf(grav[0] * grav[0] + grav[1] * grav[1] + grav[2] * grav[2]);
  for (int i = 0; i < 3; i++) grav[i] /= n;
  return true;
}
static void imu_measure_bias() {
  float s[3] = {0, 0, 0}, g[3], a[3]; int n = 0;
  for (int k = 0; k < 200; k++) { if (imu_raw(g, a)) { for (int i = 0; i < 3; i++) s[i] += g[i]; n++; } delay(5); }
  for (int i = 0; i < 3; i++) gyro_bias[i] = n ? s[i] / n : 0;
  // zwaartekracht meteen goed zetten
  float as[3]; float gs[3]; imu_raw(gs, as); map3(M_ACC, as, acc_b); level3(acc_b);
  float an = sqrtf(acc_b[0] * acc_b[0] + acc_b[1] * acc_b[1] + acc_b[2] * acc_b[2]);
  if (an > 0.1f) for (int i = 0; i < 3; i++) grav[i] = -acc_b[i] / an;
}
// opslag (EEPROM-emulatie in flash)
static void imu_load() {
  EEPROM.begin(128);
  if (EEPROM.read(63) == 0x1E) EEPROM.get(64, LEVEL);   // waterpas-correctie (2026-11-18)
  if (EEPROM.read(0) == 0xD7) { for (int i = 0; i < 9; i++) { M_ACC[i] = (int8_t)EEPROM.read(1 + i); M_GYR[i] = (int8_t)EEPROM.read(10 + i); } return; }
  int8_t I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};   // nog niet geijkt: sensor = romp
  memcpy(M_ACC, I, 9); memcpy(M_GYR, I, 9);
}
static void imu_save() {
  EEPROM.write(0, 0xD7);
  for (int i = 0; i < 9; i++) { EEPROM.write(1 + i, (uint8_t)M_ACC[i]); EEPROM.write(10 + i, (uint8_t)M_GYR[i]); }
  EEPROM.write(63, 0x1E); EEPROM.put(64, LEVEL);
  EEPROM.commit();
}
static void snap(const float v[3], int8_t row[3]) {   // naar de dichtstbijzijnde as met teken
  int k = 0; for (int i = 1; i < 3; i++) if (fabsf(v[i]) > fabsf(v[k])) k = i;
  row[0] = row[1] = row[2] = 0; row[k] = v[k] > 0 ? 1 : -1;
}
static void wait_enter(const char* msg) {
  Serial.println(msg);
  while (Serial.available()) Serial.read();
  while (!Serial.available()) delay(5);
  delay(20); while (Serial.available()) Serial.read();
}
static void mean_acc(float m[3]) {
  float g[3], a[3]; m[0] = m[1] = m[2] = 0; int n = 0;
  for (int k = 0; k < 100; k++) { if (imu_raw(g, a)) { for (int i = 0; i < 3; i++) m[i] += a[i]; n++; } delay(5); }
  for (int i = 0; i < 3; i++) m[i] /= n;
}
static void integrate_gyro_until_enter(float I[3]) {   // som van de gyro (sensor-assen) tot Enter
  float g[3], a[3]; I[0] = I[1] = I[2] = 0;
  while (Serial.available()) Serial.read();
  uint32_t t = micros();
  while (!Serial.available()) {
    if (imu_raw(g, a)) { uint32_t n = micros(); float dt = (n - t) * 1e-6f; t = n; for (int i = 0; i < 3; i++) I[i] += (g[i] - gyro_bias[i]) * dt; }
    delay(5);
  }
  delay(20); while (Serial.available()) Serial.read();
}
// 2026-11-18: IMU-ijking met 1x Enter. Montage is bekend: IMU-x = neus van de eend, IMU-z verticaal
// (omhoog of omlaag wordt gemeten), gyro en versnellingsmeter van de GY-85 hebben dezelfde assen.
// Gemeten: welke kant z op wijst, het gyro-nulpunt, en de kleine scheefstand (LEVEL) zodat
// 'rechtop in HOME op tafel' precies zwaartekracht (0, 0, -1) geeft - zoals in de simulatie.
static void imu_ijk() {
  Serial.println("IMU-ijking. Servo's staan vast in HOME. Zet hem RECHTOP op tafel, laat los, Enter.");
  wait_enter("(wacht tot hij stil staat, dan Enter)");
  float up[3]; mean_acc(up);                     // sensor-assen; wijst in rust OMHOOG
  float n = sqrtf(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
  if (n < 0.7f || n > 1.3f || fabsf(up[2]) < 0.8f * n) {
    Serial.printf("  Vreemde meting (%.2f %.2f %.2f g): staat hij stil en rechtop, zit z verticaal? NIET bewaard.\n", up[0], up[1], up[2]);
    imu_load(); return;
  }
  int8_t sz = up[2] > 0 ? 1 : -1;                // z omhoog: romp = sensor; z omlaag: y en z omgekeerd
  int8_t M[9] = {1, 0, 0, 0, sz, 0, 0, 0, sz};
  memcpy(M_ACC, M, 9); memcpy(M_GYR, M, 9);
  float a[3]; map3(M_ACC, up, a);
  for (int i = 0; i < 3; i++) a[i] /= n;         // a = 'omhoog' in romp-assen, hoort (0,0,1) te zijn
  // rotatie LEVEL die a naar (0,0,1) draait (Rodrigues)
  float v0 = a[1], v1 = -a[0], c = a[2], k = 1.0f / (1.0f + c);
  float L[9] = {1 - v1 * v1 * k, v0 * v1 * k, v1,
                v0 * v1 * k, 1 - v0 * v0 * k, -v0,
                -v1, v0, 1 - (v0 * v0 + v1 * v1) * k};
  memcpy(LEVEL, L, sizeof(L));
  Serial.printf("  IMU-z wijst %s, scheefstand %.1f gr (voorover %+.1f / links omlaag %+.1f) weggewerkt.\n", sz > 0 ? "OMHOOG" : "OMLAAG",
                acosf(clampf(c, -1, 1)) * 57.3f, asinf(clampf(-a[0], -1, 1)) * 57.3f, asinf(clampf(-a[1], -1, 1)) * 57.3f);
  imu_save(); imu_measure_bias();
  Serial.println("  Bewaard. Controle met 'imu': rechtop ~ (0, 0, -1); voorover kantelen x > 0; links omlaag y > 0; linksom draaien gyro z > 0.");
}

// ------------------------------------------------------------------ camera (ESP32-CAM)
static PacketReader<CamPacket, CAM_H0, CAM_H1> cam_reader;
static CamPacket cam_last; static uint32_t cam_t = 0; static uint8_t rp_seq = 0;
static bool cam_wifi = true; static float head_yaw_prev = 0; static uint8_t cam_lamp = 0;
static void cam_poll() { CamPacket p; while (Serial1.available()) if (cam_reader.feed((uint8_t)Serial1.read(), p)) { cam_last = p; cam_t = millis(); } }
static void cam_send() {   // 50 Hz: kopstand + draaien van het lijf + wifi-wens naar de camera
  RpPacket p; p.h0 = RP_H0; p.h1 = RP_H1; p.seq = rp_seq++;
  float yaw = q_out[HYAW], elev = q_out[HPITCH] - q_out[NECK];
  float rate = (yaw - head_yaw_prev) / DT_CTRL; head_yaw_prev = yaw;
  p.flags = (fabsf(rate) < 0.1f ? 1 : 0) | (cam_wifi ? 2 : 0);
  p.head_yaw_mrad = (int16_t)(yaw * 1000); p.head_elev_mrad = (int16_t)(elev * 1000);
  p.head_yaw_rate_mrad_s = (int16_t)clampf(rate * 1000, -32000, 32000);
  p.gyro_z_mrad_s = (int16_t)clampf(gyro_b[2] * 1000, -32000, 32000);
  p.lamp = cam_lamp;
  p.sum = duck_sum(&p, sizeof(p) - 1);
  Serial1.write((const uint8_t*)&p, sizeof(p));
}

// ------------------------------------------------------------------ servo's
static void servo_write_all(const float q[NJ], bool stagger) {
  float us[N_SERVO];
  joints_to_us(q, 0.0f, us);
  for (int n = 0; n < N_SERVO; n++) {
    pwm_set_gpio_level(n, (uint16_t)(us[n] + 0.5f));
    if (stagger && !servo_on[n]) delay(ENABLE_STAGGER_MS);   // niet alles tegelijk: stroompiek
    servo_on[n] = true;
  }
}
static void servo_limp() { for (int n = 0; n < N_SERVO; n++) { pwm_set_gpio_level(n, 0); servo_on[n] = false; } }
// 2026-11-18: servo-PWM rechtstreeks op de hardware: 50 Hz, 1 tel = 1 us. (analogWriteFreq weigert < 100 Hz
// en bleef dan op 1 kHz staan -> onzinpulsen.) Zelfde pulsfrequentie als ijk.py.
static void servo_pwm_init() {
  float div = clock_get_hz(clk_sys) / 1000000.0f;   // 1 MHz tellerklok
  for (int n = 0; n < N_SERVO; n++) {
    gpio_set_function(n, GPIO_FUNC_PWM);
    uint sl = pwm_gpio_to_slice_num(n);
    pwm_set_clkdiv(sl, div); pwm_set_wrap(sl, 19999);   // 20000 us = 50 Hz
    pwm_set_gpio_level(n, 0);
    pwm_set_enabled(sl, true);
  }
}

// ------------------------------------------------------------------ logboek (RAM, ~20 s)
struct Rec { uint32_t t; int16_t gyro[3], grav[3], vx, wz, act[NJ], q[NJ]; uint8_t mode; };
#define LOG_N 1000
static Rec logbuf[LOG_N];
static int log_i = 0, log_n = 0;
static bool log_frozen = false;
static void log_add() {
  if (log_frozen) return;
  Rec& r = logbuf[log_i];
  r.t = millis();
  for (int i = 0; i < 3; i++) { r.gyro[i] = (int16_t)(gyro_b[i] * 1000); r.grav[i] = (int16_t)(grav[i] * 10000); }
  r.vx = (int16_t)(tw[0] * 10000); r.wz = (int16_t)(tw[2] * 1000);
  for (int j = 0; j < NJ; j++) { r.act[j] = (int16_t)(duck.act[j] * 1000); r.q[j] = (int16_t)(q_out[j] * 1000); }
  r.mode = (uint8_t)mode;
  log_i = (log_i + 1) % LOG_N; if (log_n < LOG_N) log_n++;
}
static void log_dump() {
  Serial.print("t_ms,modus,gx,gy,gz,zx,zy,zz,vx,wz");
  for (int j = 0; j < NJ; j++) Serial.printf(",a_%s", J_NAMES[j]);
  for (int j = 0; j < NJ; j++) Serial.printf(",q_%s", J_NAMES[j]);
  Serial.println();
  int s = (log_i - log_n + LOG_N) % LOG_N;
  for (int k = 0; k < log_n; k++) {
    Rec& r = logbuf[(s + k) % LOG_N];
    Serial.printf("%lu,%s,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.3f", (unsigned long)r.t, MODE_NAAM[r.mode],
                  r.gyro[0] / 1000.f, r.gyro[1] / 1000.f, r.gyro[2] / 1000.f, r.grav[0] / 1e4f, r.grav[1] / 1e4f, r.grav[2] / 1e4f,
                  r.vx / 1e4f, r.wz / 1e3f);
    for (int j = 0; j < NJ; j++) Serial.printf(",%.3f", r.act[j] / 1000.f);
    for (int j = 0; j < NJ; j++) Serial.printf(",%.3f", r.q[j] / 1000.f);
    Serial.println();
  }
  Serial.println("# einde log");
}

// ------------------------------------------------------------------ modi
static void set_mode(Mode m) {
  if (m == NET && mode != NET) {    // 2026-11-18: niet starten als de IMU zegt dat hij scheef staat
    float tilt0 = acosf(clampf(-grav[2], -1, 1)) * 57.3f;
    if (tilt0 > START_MAX_DEG) {
      Serial.printf("NIET gestart: IMU meet %.0f gr kanteling (max %.0f). Staat hij rechtop? Anders eerst 'imu ijk'.\n", tilt0, START_MAX_DEG);
      tw_goal[0] = tw_goal[2] = 0;
      return;
    }
  }
  if (m != THUIS) wip_deg = 0;
  if (m == SLAP) servo_limp();
  if (m != NET) { oefen_left = 0; steps_left = 0; tw_stop_ms = 0; step_wait = false; last_half = -1; }   // geen schema meer na slap/thuis/val
  if (mode == NET && m != NET) log_frozen = true;                  // log van de netwerk-run bewaren (ook na slap/knop)
  if (m == NET && mode != NET) { log_n = 0; log_i = 0; log_frozen = false; }   // nieuwe run: nieuw log
  if ((m == THUIS || m == ANIM || m == NET) && mode == SLAP) {
    for (int j = 0; j < NJ; j++) q_out[j] = 0;
    servo_write_all(q_out, true);   // vanuit slap: een voor een aan, naar HOME
  }
  if (m == NET && mode != NET) {    // netwerk start altijd vanuit HOME (zoals de training)
    for (int j = 0; j < NJ; j++) q_out[j] = 0;
    servo_write_all(q_out, false);
    delay(mode == THUIS ? 300 : 1000);
    duck.reset(); tw[0] = tw[1] = tw[2] = 0;
    post = STAAT; post_t = 0; sit_amt = 0; rest_b = 0;
  }
  if (m == ANIM && mode != ANIM) { anim_a = 0; anim_ph = 0; }
  mode = m; mode_t0 = millis();
  Serial.printf("modus: %s\n", MODE_NAAM[m]);
}
static void status() {
  Serial.printf("houding %s | rust %s | netwerk-aandeel %.0f%%\n", POST_NAAM[post], rust_aan ? "aan" : "uit", rest_b * 100);
  Serial.printf("modus %s | netwerk %s | opdracht vx %.3f m/s wz %.0f gr/s | accu %.2f V\n", MODE_NAAM[mode], POLICY_NAAM,
                tw[0], tw[2] * 57.3f, bat_v);
  Serial.printf("zwaartekracht %.2f %.2f %.2f (kanteling %.0f gr) | gyro %.2f %.2f %.2f rad/s\n", grav[0], grav[1], grav[2],
                acosf(clampf(-grav[2], -1, 1)) * 57.3f, gyro_b[0], gyro_b[1], gyro_b[2]);
  Serial.printf("netwerk %.2f ms/stap | langste lus %.2f ms | log %d stappen%s\n", t_infer_us / 1000, t_loop_max_us / 1000, log_n,
                log_frozen ? " (BEVROREN na val)" : "");
  Serial.printf("camera: %s (goed %lu, fout %lu) | wifi camera %s\n", (cam_t && millis() - cam_t < 500) ? "verbonden" : "GEEN",
                (unsigned long)cam_reader.ok, (unsigned long)cam_reader.bad, cam_wifi ? "aan" : "uit");
  Serial.printf("nog niet geijkt: %s\n", CAL_NOG_NIET);
}
// 2026-11-18: controleert of de gyro dezelfde kant op draait als de versnellingsmeter (de 1-Enter-ijking neemt aan
// dat ze dezelfde assen hebben). Kantelen meet de versnellingsmeter rechtstreeks; de gyro opgeteld moet hetzelfde geven.
// Verkeerd teken -> die gyro-as wordt omgedraaid en bewaard.
static void imu_test() {
  Serial.println("IMU-TEST (12 s). Houd hem in je hand en doe rustig, ~1 s per beweging:");
  Serial.println("  1) VOOROVER kantelen (~30 gr) en terug   2) LINKERKANT OMLAAG (~30 gr) en terug   3) LINKSOM draaien ~90 gr");
  Serial.println("Start over 2 s...");
  delay(2000);
  imu_measure_bias();
  float af = 0, al = 0, gf = 0, gl = 0, gz = 0, sff = 0, sgg = 0, sfg = 0, sll = 0, shh = 0, slh = 0;
  uint32_t t0 = millis(), tp = 0, tl = micros();
  while (millis() - t0 < 12000) {
    if (!imu_update(0.005f)) continue;
    uint32_t n = micros(); float dt = (n - tl) * 1e-6f; tl = n;
    float an = sqrtf(acc_b[0] * acc_b[0] + acc_b[1] * acc_b[1] + acc_b[2] * acc_b[2]);
    if (an > 0.1f) { af = asinf(clampf(-acc_b[0] / an, -1, 1)) * 57.3f; al = asinf(clampf(-acc_b[1] / an, -1, 1)) * 57.3f; }
    gf += gyro_b[1] * dt * 57.3f;     // voorover = draaien om +y
    gl += -gyro_b[0] * dt * 57.3f;    // linkerkant omlaag = draaien om -x
    gz += gyro_b[2] * dt * 57.3f;     // linksom = +z
    sff += af * af; sgg += gf * gf; sfg += af * gf; sll += al * al; shh += gl * gl; slh += al * gl;
    if (millis() - tp > 300) { tp = millis();
      Serial.printf("  voorover: versn %+5.0f  gyro %+5.0f   | links omlaag: versn %+5.0f  gyro %+5.0f   | linksom gyro %+5.0f gr\n", af, gf, al, gl, gz); }
    delay(5);
  }
  float cf = sfg / sqrtf(sff * sgg + 1e-6f), cl = slh / sqrtf(sll * shh + 1e-6f);
  bool fl_y = cf < -0.5f, fl_x = cl < -0.5f, fl_z = gz < -30.0f;
  Serial.printf("  voorover (gyro y): %s (samenhang %+.2f)\n", fl_y ? "OMGEKEERD" : (cf > 0.5f ? "goed" : "?? te weinig bewogen"), cf);
  Serial.printf("  links omlaag (gyro x): %s (samenhang %+.2f)\n", fl_x ? "OMGEKEERD" : (cl > 0.5f ? "goed" : "?? te weinig bewogen"), cl);
  Serial.printf("  linksom (gyro z): %+.0f gr -> %s\n", gz, fl_z ? "OMGEKEERD" : (gz > 30.0f ? "goed" : "?? te weinig gedraaid"));
  if (fl_x || fl_y || fl_z) {
    for (int i = 0; i < 3; i++) { if (fl_x) M_GYR[i] = -M_GYR[i]; if (fl_y) M_GYR[3 + i] = -M_GYR[3 + i]; if (fl_z) M_GYR[6 + i] = -M_GYR[6 + i]; }
    imu_save(); imu_measure_bias();
    Serial.println("  Gyro-assen omgedraaid en bewaard. Doe 'imu test' nog een keer: alles moet nu 'goed' zijn.");
  }
}

static void command(String s) {
  s.trim(); if (!s.length()) return;
  int sp = s.indexOf(' '); String c = sp < 0 ? s : s.substring(0, sp); String r = sp < 0 ? "" : s.substring(sp + 1); r.trim();
  if (c == "slap" || c == "q") set_mode(SLAP);
  else if (c == "thuis") { wip_deg = 0; for (int j = 0; j < NJ; j++) q_zet[j] = 0; set_mode(THUIS); }
  else if (c == "wip") {   // wip [gr]: ALLE gewrichten tegelijk heen en weer tussen HOME en +gr (tekencontrole)
    wip_deg = clampf(r.length() ? r.toFloat() : 10.0f, -15, 15);
    if (mode != THUIS) { for (int j = 0; j < NJ; j++) q_zet[j] = 0; set_mode(THUIS); }
    wip_t0 = millis();
    Serial.printf("wip: alle gewrichten HOME <-> %+.0f gr. 'thuis' = stoppen.\n", wip_deg);
  }
  else if (c == "zet") {   // zet <gewricht> <graden>: tekencontrole, langzaam, vanuit HOME
    int s2 = r.indexOf(' ');
    String jn = s2 < 0 ? r : r.substring(0, s2); float deg = s2 < 0 ? 0 : r.substring(s2 + 1).toFloat();
    int k = -1;
    for (int j = 0; j < NJ; j++) if (jn == J_NAMES[j]) k = j;
    if (k < 0) { Serial.print("onbekend gewricht. Kies uit:"); for (int j = 0; j < NJ; j++) Serial.printf(" %s", J_NAMES[j]); Serial.println(); }
    else {
      wip_deg = 0;
      if (mode != THUIS) { for (int j = 0; j < NJ; j++) q_zet[j] = 0; set_mode(THUIS); }
      q_zet[k] = clampf(deg / 57.3f, J_LO[k], J_HI[k]);
      Serial.printf("zet %s naar %+.0f gr (t.o.v. HOME). 'thuis' = alles terug.\n", J_NAMES[k], q_zet[k] * 57.3f);
    }
  }
  else if (c == "anim") { anim_amp = r.length() ? clampf(r.toFloat(), 0, 1) : 1.0f; set_mode(ANIM); }
  else if (c == "staan" || c == "opstaan") { steps_left = 0; oefen_left = 0; tw_goal[0] = tw_goal[2] = 0; want_sit = false; if (mode != NET) set_mode(NET); }
  else if (c == "zit") { tw_goal[0] = tw_goal[2] = 0; want_sit = true; if (mode != NET) set_mode(NET); }
  else if (c == "rust") {   // rust aan (standaard, = player): stilstaan = HOME vasthouden. rust uit: netwerk ook bij stilstaan
    if (r == "aan" || r == "uit") rust_aan = (r == "aan");
    Serial.printf("rust: %s\n", rust_aan ? "AAN (stilstaan = HOME, netwerk alleen bij lopen)" : "UIT (netwerk ook bij stilstaan = balanceren)");
  }
  else if (c == "oefen") {   // oefen [k] [n] [f] [rust]: k keer n passen op f, met rust ertussen (standaard 5 1 0.25 1.5)
    float a[4] = {5, 1, 0.25f, 1.5f}; int i = 0; String rest = r; rest.trim();
    while (rest.length() && i < 4) { int sp = rest.indexOf(' '); a[i++] = (sp < 0 ? rest : rest.substring(0, sp)).toFloat(); rest = sp < 0 ? "" : rest.substring(sp + 1); rest.trim(); }
    oefen_n = max(1, (int)a[1]); oefen_f = clampf(a[2], 0.0f, 1.0f); oefen_rust = clampf(a[3], 0.3f, 10.0f);
    oefen_left = max(1, (int)a[0]); tw_stop_ms = 0; want_sit = false;
    if (mode != NET) set_mode(NET);
    oefen_t = millis() + 1000;   // eerst 1 s staan
    Serial.printf("oefen: %d keer %d pas(sen) op %.0f %%, %.1f s rust ertussen\n", oefen_left, oefen_n, oefen_f * 100, oefen_rust);
  }
  else if (c == "passen") {
    oefen_left = 0;   // passen <n> [f]: n passen vooruit op f x 100% (standaard 0.3), dan staan
    int sp = r.indexOf(' ');
    int n = (sp < 0 ? r : r.substring(0, sp)).toInt(); float f = sp < 0 ? 0.3f : r.substring(sp + 1).toFloat();
    if (n <= 0) { Serial.println("gebruik: passen <n> [f], bv. passen 3"); return; }
    steps_left = n; tw_stop_ms = 0; want_sit = false;
    step_wait = true; step_v = clampf(f, 0.0f, 1.0f) * V100; tw_goal[0] = tw_goal[2] = 0;
    if (mode != NET) set_mode(NET);
    step_ph = duck.phase;
    Serial.printf("passen: %d stuks op %.0f %%\n", n, f * 100);
  }
  else if (c == "loop" || c == "draai") {
    steps_left = 0; oefen_left = 0;   // loop <f> [w] [s] | draai <w> [s]   (s = duur in seconden, 0/leeg = blijven)
    float a[3] = {0, 0, 0}; int n = 0; String rest = r; rest.trim();
    while (rest.length() && n < 3) {
      int sp = rest.indexOf(' ');
      a[n++] = (sp < 0 ? rest : rest.substring(0, sp)).toFloat();
      rest = sp < 0 ? "" : rest.substring(sp + 1); rest.trim();
    }
    float f = c == "loop" ? a[0] : 0, w = c == "loop" ? a[1] : a[0], dur = c == "loop" ? a[2] : a[1];
    tw_goal[0] = clampf(f, -0.5f, 1.0f) * V100; tw_goal[2] = clampf(w, -90, 90) / 57.3f; want_sit = false;
    tw_stop_ms = dur > 0 ? millis() + (uint32_t)(dur * 1000) : 0;
    if (mode != NET) set_mode(NET);
    if (dur > 0) Serial.printf("%s: vx %.3f m/s, wz %.0f gr/s, %.1f s - daarna staan\n", c.c_str(), tw_goal[0], tw_goal[2] * 57.3f, dur);
    else Serial.printf("%s: vx %.3f m/s, wz %.0f gr/s\n", c.c_str(), tw_goal[0], tw_goal[2] * 57.3f);
  }
  else if (c == "imu") {
    if (r == "ijk") { for (int j = 0; j < NJ; j++) q_zet[j] = 0; set_mode(THUIS); imu_ijk(); }   // servo's vast in HOME = staande houding
    else if (r == "nul") { imu_measure_bias(); Serial.println("gyro-nulpunt gemeten"); }
    else if (r == "test") { set_mode(SLAP); imu_test(); }
    else {
      for (int k = 0; k < 25; k++) {
        Serial.printf("zwaartekracht %+.2f %+.2f %+.2f | gyro %+.2f %+.2f %+.2f rad/s\n", grav[0], grav[1], grav[2], gyro_b[0], gyro_b[1], gyro_b[2]);
        uint32_t t = millis(); while (millis() - t < 200) { imu_update(0.005f); delay(5); }
      }
    }
  }
  else if (c == "log") { if (r == "wis") { log_n = 0; log_i = 0; log_frozen = false; Serial.println("log gewist"); } else log_dump(); }
  else if (c == "status") status();
  else if (c == "wifi") {
    if (r == "aan" || r == "uit") { cam_wifi = (r == "aan"); EEPROM.write(30, 0xC1); EEPROM.write(31, cam_wifi); EEPROM.commit(); }
    Serial.printf("wifi camera: %s (bewaard)\n", cam_wifi ? "AAN" : "UIT");
  }
  else if (c == "lamp") { cam_lamp = (uint8_t)clampf(r.toFloat(), 0, 100); Serial.printf("lamp camera %d %%\n", cam_lamp); }
  else if (c == "cam") {
    for (int k = 0; k < 15; k++) {
      uint32_t t = millis(); while (millis() - t < 200) { cam_poll(); delay(2); }
      CamPacket& p = cam_last;
      if (!cam_t || millis() - cam_t > 500) { Serial.printf("geen camera (goed %lu, fout %lu pakketjes)\n", (unsigned long)cam_reader.ok, (unsigned long)cam_reader.bad); continue; }
      Serial.printf("ziet %s | richting %+4.0f / %+4.0f gr | afstand %.2f m | grootte %d | intensiteit %d | eigen bew. %+.2f rad/s | nadering %+.2f m/s | %d fps\n",
                    (p.flags & 1) ? "JA " : "nee", p.bx_mrad * 0.0573f, p.by_mrad * 0.0573f, p.dist_mm / 1000.f, p.msize, p.mint,
                    p.tflow_mrad_s / 1000.f, p.approach_mm_s / 1000.f, p.fps);
    }
  }
  else Serial.println("onbekend commando. slap, thuis, zit, staan, rust aan|uit, imu test, zet <gewricht> <gr>, wip [gr], anim [a], staan, loop <f> [w], draai <w>, imu [ijk|nul], log [wis], status, cam, wifi aan|uit, lamp <0-100>");
}

// ------------------------------------------------------------------ setup / loop
void setup() {
  Serial.begin(115200);
  pinMode(PIN_BTN, INPUT_PULLUP);
  servo_pwm_init();   // 50 Hz, 1 us per tel
  servo_limp();
  analogReadResolution(12);
  pol.init(); duck.reset();
  imu_init(); imu_load();
  Serial1.setTX(PIN_CAM_TX); Serial1.setRX(PIN_CAM_RX); Serial1.begin(CAM_BAUD);
  if (EEPROM.read(30) == 0xC1) cam_wifi = EEPROM.read(31);
  delay(300);
  imu_measure_bias();
  uint32_t t = millis(); while (!Serial && millis() - t < 2000) delay(10);
  Serial.printf("Duckling loop-firmware | netwerk %s | servo's SLAP. Typ 'status' of 'thuis'.\n", POLICY_NAAM);
  if (EEPROM.read(0) != 0xD7) Serial.println("LET OP: IMU nog niet geijkt -> eerst 'imu ijk'.");
}

void loop() {
  static uint32_t next = micros(); static int tick = 0; static String line;
  static uint32_t btn_t = 0; static bool btn_prev = false;
  cam_poll();
  while (Serial.available()) { char ch = Serial.read(); if (ch == '\n' || ch == '\r') { command(line); line = ""; } else line += ch; }

  // knopje: kort = noodstop, lang (2 s) vanuit slap = staan
  bool b = digitalRead(PIN_BTN) == LOW;
  if (b && !btn_prev) { btn_t = millis(); if (mode != SLAP) set_mode(SLAP); }
  if (b && mode == SLAP && btn_t && millis() - btn_t > 2000) { btn_t = 0; tw_goal[0] = tw_goal[2] = 0; want_sit = false; set_mode(NET); }
  btn_prev = b;

  if ((int32_t)(micros() - next) < 0) return;
  uint32_t t0 = micros();
  next += 1000000 / IMU_HZ;
  if ((int32_t)(t0 - next) > 20000) next = t0;   // achterstand: niet inhalen
  imu_update(1.0f / IMU_HZ);
  if (++tick < CTRL_EVERY) return;
  tick = 0;

  // ---- 50 Hz
  float tilt = acosf(clampf(-grav[2], -1, 1)) * 57.3f;
  if ((mode == NET || mode == ANIM) && tilt > FALL_DEG) {
    set_mode(SLAP); log_frozen = true;
    Serial.printf("GEVALLEN (kanteling %.0f gr) -> slap. 'log' laat de seconden ervoor zien.\n", tilt);
  }
  static int bat_k = 0;
  if (++bat_k >= 50) {
    bat_k = 0; bat_v = analogRead(PIN_BAT) * 3.3f / 4095.0f * 2.0f;
    if (bat_v > 2.5f && bat_v < BAT_MIN) { if (!bat_low_t) bat_low_t = millis(); else if (millis() - bat_low_t > 3000 && mode != SLAP) { set_mode(SLAP); Serial.println("ACCU LEEG -> slap"); } }
    else bat_low_t = 0;
  }
  if (mode == THUIS) {   // naar q_zet (HOME of 'zet'), max 60 gr/s gewricht
    if (wip_deg != 0) {    // wip: 1,5 s op +wip_deg, 1,5 s op HOME, herhalen
      bool op = ((millis() - wip_t0) / 1500) % 2 == 0;
      for (int j = 0; j < NJ; j++) q_zet[j] = op ? clampf(wip_deg / 57.3f, J_LO[j], J_HI[j]) : 0;
    }
    for (int j = 0; j < NJ; j++) q_out[j] += clampf(q_zet[j] - q_out[j], -1.047f * DT_CTRL, 1.047f * DT_CTRL);
    servo_write_all(q_out, false);
  } else if (mode == ANIM) {   // open lus: loopje, amplitude in 2 s omhoog
    anim_a = clampf(anim_a + DT_CTRL / 2.0f, 0, anim_amp);
    anim_ph += DT_CTRL / ANIM_CYCLE; anim_ph -= floorf(anim_ph);
    anim_pose(anim_ph, anim_a, q_out);
    servo_write_all(q_out, false);
  } else if (mode == NET) {
    if (steps_left > 0 && rest_b > 0.5f) {   // een pas = een halve gangcyclus (fase door 0 of 0.5)
      float ph = duck.phase;
      int h = (int)floorf(ph * 2);
      bool crossed = h != (int)floorf(step_ph * 2) || ph < step_ph;
      if (step_wait) {   // = training: pas beginnen op de grens (gewicht verdeeld), met de andere voet dan de vorige keer
        if (crossed && h != last_half) { step_wait = false; tw_goal[0] = step_v; tw_goal[2] = 0; }
      } else if (crossed) {
        steps_left--;
        Serial.printf("pas (nog %d)\n", steps_left);
        if (steps_left == 0) {
          last_half = 1 - h;
          tw_goal[0] = tw_goal[2] = 0;
          if (oefen_left > 0) { oefen_t = millis() + (uint32_t)(oefen_rust * 1000); Serial.printf("rust (nog %d keer)\n", oefen_left); }
          else Serial.println("passen klaar: staan");
        }
      }
      step_ph = ph;
    } else step_ph = duck.phase;
    if (oefen_left > 0 && steps_left == 0 && oefen_t && (int32_t)(millis() - oefen_t) >= 0) {   // volgende ronde
      oefen_t = 0; oefen_left--; steps_left = oefen_n; step_ph = duck.phase;
      step_wait = true; step_v = oefen_f * V100; tw_goal[0] = tw_goal[2] = 0;
    }
    if (tw_stop_ms && (int32_t)(millis() - tw_stop_ms) >= 0) {
      tw_stop_ms = 0; tw_goal[0] = tw_goal[2] = 0; Serial.println("tijd om: staan");
    }
    // houding (= duckling_gedrag.py)
    post_t += DT_CTRL;
    if (post == STAAT && want_sit && rest_b <= 0.0f) { post = GAAT_ZITTEN; post_t = 0; Serial.println("houding: GAAT ZITTEN"); }
    else if (post == ZIT && !want_sit) { post = STAAT_OP; post_t = 0; Serial.println("houding: STAAT OP"); }
    if (post == GAAT_ZITTEN) { sit_amt = min_jerk(post_t / SIT_S); if (post_t >= SIT_S) { post = ZIT; Serial.println("houding: ZIT"); } }
    else if (post == STAAT_OP) { sit_amt = 1.0f - min_jerk(post_t / SIT_UP_S); if (post_t >= SIT_UP_S) { post = STAAT; sit_amt = 0; Serial.println("houding: STAAT"); } }
    bool standing = post == STAAT;
    float goal_v = standing && !want_sit ? tw_goal[0] : 0.0f, goal_w = standing && !want_sit ? tw_goal[2] : 0.0f;
    tw[0] += clampf(goal_v - tw[0], -ACC_V * DT_CTRL, ACC_V * DT_CTRL);
    tw[2] += clampf(goal_w - tw[2], -ACC_W * DT_CTRL, ACC_W * DT_CTRL);
    if (!standing) tw[0] = tw[2] = 0;
    bool moving = fabsf(tw[0]) > 0.004f || fabsf(tw[2]) > 0.1f;
    if (!rust_aan && standing) rest_b = 1.0f;
    else rest_b = clampf(rest_b + (moving ? DT_CTRL / FADE_IN_S : -DT_CTRL / FADE_OUT_S), 0, 1);
    uint32_t a = micros();
    float act[NJ];
    if (standing) { duck.build_obs(gyro_b, grav, tw, head_cmd); pol.run(duck.obs, duck.act); memcpy(act, duck.act, sizeof(act)); }
    else for (int j = 0; j < NJ; j++) act[j] = 0;
    float bl[NJ], offs[NJ];   // 2026-11-26: netwerk (gewogen) en zithouding apart: bij residu-acties is de actie
    for (int j = 0; j < NJ; j++) { bl[j] = rest_b * act[j]; offs[j] = (1 - rest_b) * sit_amt * CROUCH[j]; }   // een correctie
    duck.apply(bl, tw, head_cmd, offs);
    t_infer_us = micros() - a;
    memcpy(q_out, duck.filt, sizeof(q_out));
    servo_write_all(q_out, false);
  }
  cam_send();
  if (mode == NET || mode == ANIM) log_add();   // alleen netwerk/anim (THUIS overschreef het log)
  float dt = micros() - t0; if (dt > t_loop_max_us) t_loop_max_us = dt;
}
