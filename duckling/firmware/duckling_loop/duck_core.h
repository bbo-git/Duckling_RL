// duck_core.h - rekenkern van de Duckling-firmware, zonder Arduino-afhankelijkheden (2026-11-02).
// Exact zoals de training / tools/duckling_play_onnx.py:
//   observatie (65) -> normaliseren -> MLP 65-256-128-64-14 (ELU) -> actie
//   doel = HOME + actie, geclipt op het gewrichtsbereik -> laagdoorlaat (alpha 0.4, kop 0.25)
//   -> geschatte gewrichtsstand (geen encoders: max 0.75 x servosnelheid per stap)
//   -> servo's (differentieel: A = 1.5(sp*uit + sr*drager), B = 1.5(sp*uit - sr*drager)).
// Wordt op de pc getest tegen onnxruntime (tools/test_core.cpp).
#pragma once
#include <math.h>
#include <string.h>
#include "policy_weights.h"
#include "duck_data.h"
#include "calib.h"

#define NJ 14
#define DT_CTRL 0.02f            // 50 Hz, = training
#define US_MID 1500.0f
#define US_PER_DEG 10.0f         // = ijk.py
#define US_MIN 600.0f
#define US_MAX 2400.0f
#define RATIO 1.5f
#define N_SERVO 15               // GPIO 0..14

// gewricht-indexen (trainingsvolgorde)
enum { LHY, LHR, LHP, LKN, LAN, NECK, HPITCH, HROLL, HYAW, RHY, RHR, RHP, RKN, RAN };

static inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ------------------------------------------------------------------ netwerk
struct Policy {
  float w0[L0_OUT * L0_IN], w1[L1_OUT * L1_IN], w2[L2_OUT * L2_IN], w3[L3_OUT * L3_IN];
  void init() {   // gewichten van flash naar RAM (sneller dan uit flash lezen)
    memcpy(w0, L0_W, sizeof(w0)); memcpy(w1, L1_W, sizeof(w1));
    memcpy(w2, L2_W, sizeof(w2)); memcpy(w3, L3_W, sizeof(w3));
  }
  static void dense(const float* W, const float* B, const float* x, float* y, int nin, int nout, bool elu) {
    for (int o = 0; o < nout; o++) {
      const float* w = W + o * nin;
      float s = B[o];
      for (int i = 0; i < nin; i++) s += w[i] * x[i];
      y[o] = (elu && s < 0.0f) ? expf(s) - 1.0f : s;
    }
  }
  void run(const float* obs, float* act) {
    float x[N_OBS], h0[L0_OUT], h1[L1_OUT], h2[L2_OUT];
    for (int i = 0; i < N_OBS; i++) x[i] = (obs[i] - OBS_MEAN[i]) / OBS_STD[i];
    dense(w0, L0_B, x, h0, L0_IN, L0_OUT, true);
    dense(w1, L1_B, h0, h1, L1_IN, L1_OUT, true);
    dense(w2, L2_B, h1, h2, L2_IN, L2_OUT, true);
    dense(w3, L3_B, h2, act, L3_IN, L3_OUT, false);
  }
};

// ------------------------------------------------------------------ gangritme (= GridClock)
static inline void grid_axis(const float* ax, int n, float x, int* i, float* w) {
  x = clampf(x, ax[0], ax[n - 1]);
  int k = 0;
  while (k < n - 2 && x >= ax[k + 1]) k++;   // = clip(searchsorted(ax, x) - 1, 0, n - 2)
  *i = k; *w = (x - ax[k]) / (ax[k + 1] - ax[k]);
}
static inline float grid_cycle(float vx, float wz) {
  int i, j; float wi, wj;
  grid_axis(GRID_SPEEDS, N_SPEEDS, vx / V100, &i, &wi);
  grid_axis(GRID_TURNS_DEG, N_TURNS, wz * 57.29578f, &j, &wj);
  const float* c = GRID_CYCLE;
  #define C(a, b) c[(a) * N_TURNS + (b)]
  return (1 - wi) * ((1 - wj) * C(i, j) + wj * C(i, j + 1)) + wi * ((1 - wj) * C(i + 1, j) + wj * C(i + 1, j + 1));
  #undef C
}
static inline float grid_walk(float vx, float vy, float wz) {   // 0 = staan, 1 = lopen (klok vervaagt)
  float s = fabsf(vx / V100) / 0.15f, t = fabsf(wz * 57.29578f) / 30.0f, y = fabsf(vy) * 0.45f / 0.006f;
  float m = s > t ? s : t; m = m > y ? m : y;
  return clampf(m, 0.0f, 1.0f);
}

// ------------------------------------------------------------------ toestand + een stap
struct Duck {
  float filt[NJ], est[NJ], est_prev[NJ], last_action[NJ];
  float phase;
  float obs[N_OBS], act[NJ];
  int style;   // 0 = normaal, 1 = parmantig

  void reset() {   // na het naar HOME brengen: alles op HOME (V2: HOME = 0)
    for (int j = 0; j < NJ; j++) { filt[j] = est[j] = est_prev[j] = 0.0f; last_action[j] = 0.0f; }
    phase = 0.0f; style = 0;
  }
  // twist: vx (m/s), vy, wz (rad/s); head: nek, kop-pitch, kop-yaw, kop-roll (rad); gyro rad/s, grav eenheidsvector (romp-assen)
  void build_obs(const float gyro[3], const float grav[3], const float twist[3], const float head[4]) {
    float* o = obs; int k = 0;
    for (int i = 0; i < 3; i++) o[k++] = gyro[i];
    for (int i = 0; i < 3; i++) o[k++] = grav[i];
    for (int j = 0; j < NJ; j++) o[k++] = est[j];                                  // - HOME (= 0)
    for (int j = 0; j < NJ; j++) o[k++] = (est[j] - est_prev[j]) / DT_CTRL;
    for (int j = 0; j < NJ; j++) o[k++] = last_action[j];
    for (int i = 0; i < 3; i++) o[k++] = twist[i];
    bool moving = sqrtf(twist[0] * twist[0] + twist[1] * twist[1]) > 0.02f || fabsf(twist[2]) > 0.1f;
    for (int i = 0; i < 4; i++) o[k++] = moving ? head[i] : 0.0f;               // = training: geen kijkopdracht bij stilstaan
    for (int i = 0; i < 6; i++) o[k++] = 0.0f;
    float w = grid_walk(twist[0], twist[1], twist[2]);
    o[k++] = w * sinf(6.2831853f * phase);
    o[k++] = w * cosf(6.2831853f * phase);
    o[k++] = style == 0 ? 1.0f : 0.0f;
    o[k++] = style == 1 ? 1.0f : 0.0f;
  }
  // na het netwerk: actie -> servodoelen (filt). = Duckling.step
  void apply(const float* action, const float twist[3]) {
    phase += DT_CTRL / grid_cycle(twist[0], twist[2]);
    phase -= floorf(phase);
    for (int j = 0; j < NJ; j++) {
      last_action[j] = action[j];
      float t = clampf(action[j], J_LO[j], J_HI[j]);           // HOME + actie, geclipt
      filt[j] += J_ALPHA[j] * (t - filt[j]);
      est_prev[j] = est[j];
      float st = J_VEST[j] * DT_CTRL;
      est[j] += clampf(filt[j] - est[j], -st, st);
    }
  }
  void step(Policy& p, const float gyro[3], const float grav[3], const float twist[3], const float head[4]) {
    build_obs(gyro, grav, twist, head);
    p.run(obs, act);
    apply(act, twist);
  }
};

// ------------------------------------------------------------------ gewrichten -> servo-microseconden
// q: 14 gewrichtshoeken (rad, trainingsvolgorde), kaak in graden. us[15] per GPIO 0..14. = ijk.py
static inline float servo_us(int n, float servo_deg) {
  return clampf(US_MID + CAL_TEKEN[n] * (CAL_OFFSET[n] + servo_deg) * US_PER_DEG, US_MIN, US_MAX);
}
static inline void joints_to_us(const float q[NJ], float kaak_deg, float us[N_SERVO]) {
  const float D = 57.29578f;
  float d[NJ];
  for (int j = 0; j < NJ; j++) d[j] = q[j] * D;
  us[0] = servo_us(0, RATIO * (LH_SP * d[LHP] + LH_SR * d[LHR]));    // L_diffA
  us[1] = servo_us(1, RATIO * (LH_SP * d[LHP] - LH_SR * d[LHR]));    // L_diffB
  us[2] = servo_us(2, RATIO * (RH_SP * d[RHP] + RH_SR * d[RHR]));    // R_diffA
  us[3] = servo_us(3, RATIO * (RH_SP * d[RHP] - RH_SR * d[RHR]));    // R_diffB
  us[4] = servo_us(4, RATIO * (KOP_SP * d[HYAW] + KOP_SR * d[HROLL]));   // kop_diffA
  us[5] = servo_us(5, RATIO * (KOP_SP * d[HYAW] - KOP_SR * d[HROLL]));   // kop_diffB
  us[6] = servo_us(6, d[LHY]);     // L_yaw
  us[7] = servo_us(7, d[RHY]);     // R_yaw
  us[8] = servo_us(8, d[LKN]);     // L_knie
  us[9] = servo_us(9, d[RKN]);     // R_knie
  us[10] = servo_us(10, d[LAN]);   // L_enkel
  us[11] = servo_us(11, d[RAN]);   // R_enkel
  us[12] = servo_us(12, d[NECK]);  // nek
  us[13] = servo_us(13, d[HPITCH]);// kop_pitch
  us[14] = servo_us(14, kaak_deg); // kaak
}

// animatietest: loopje (open lus) op fase ph, met amplitude a (0..1) rond HOME
static inline void anim_pose(float ph, float a, float q[NJ]) {
  float f = ph * ANIM_N; int i0 = (int)f % ANIM_N; float w = f - floorf(f); int i1 = (i0 + 1) % ANIM_N;
  for (int j = 0; j < NJ; j++)
    q[j] = clampf(a * ((1 - w) * ANIM_Q[i0 * NJ + j] + w * ANIM_Q[i1 * NJ + j]), J_LO[j], J_HI[j]);
}
