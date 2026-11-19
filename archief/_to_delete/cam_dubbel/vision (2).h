// vision.h - bewegingsdetectie van de Duckling-camera (ESP32-CAM), zonder Arduino-afhankelijkheden (2026-11-03).
// Wordt op de pc getest (tools/test_vision.cpp).
//
// Per beeld (grijs 160x120 -> 80x60):
//   1. eigen beweging eruit: globale verschuiving t.o.v. het vorige beeld (kop/lijf draait) zoeken (+-R px)
//   2. verschilbeeld |nu - vorige (verschoven)|, onder de drempel = 0
//   3. bewegingsenergie per cel (16x12 cellen van 5x5 px), glijdend gemiddeld over de tijd
//   4. de sterkste cel + aaneengesloten cellen = de vlek: midden, kader, oppervlak, intensiteit
//   5. -> protocol: richting (bx, by), grootte 0..3, intensiteit 0..3, afstand, eigen beweging, nadering
#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>

#define VW 80
#define VH 60
#define CS 5                  // celgrootte (px)
#define CW (VW / CS)          // 16
#define CH (VH / CS)          // 12

struct VisionCfg {            // instelbaar via de webpagina, bewaard in flash
  float diff_thr = 14;        // grijswaardenverschil dat als beweging telt
  float min_energy = 6;       // minimale intensiteit van de vlek om iets te zien
  float i1 = 6, i2 = 14, i3 = 26;        // intensiteit (gem. verschil per pixel in de vlek) -> niveau 1/2/3
  float s1 = 0.0, s2 = 0.06, s3 = 0.18;  // oppervlak (fractie van het beeld) -> grootte 1/2/3
  float fov_h_deg = 60, fov_v_deg = 45;  // = training (FOV_H, FOV_V = halve hoeken 30 / 22 gr)
  float cam_h = 0.13;         // m: hoogte van de camera boven de vloer (staand)
  float size_k = 0.12;        // afstand uit oppervlak: d = size_k / sqrt(oppervlak) (als de onderkant niet zichtbaar is)
  float smooth = 0.4;         // 0..1: hoeveel van het nieuwe beeld in de energie (lager = rustiger, trager)
  int flip_x = 0, flip_y = 0; // camera gespiegeld gemonteerd
  int rot = 0;                // 2026-11-03: camera gedraaid gemonteerd: 0, 90, 180 of 270 gr (met de klok mee)
  int res = 0;                // opnameformaat: 0 = 160x120, 1 = 320x240 (zelfde beeldhoek verwacht; om te vergelijken)
  int shift_r = 5;            // zoekbereik eigen beweging (px op 80x60)
  int lag = 2;                // vergelijken met het beeld van zoveel beelden terug (1..3): langzame beweging zichtbaar
};

struct VisionOut {
  bool vis = false;
  float bx = 0, by = 0, dist = 1.0f, tflow = 0, approach = 0;
  int msize = 0, mint = 0;
  float area = 0, inten = 0;
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // kader (px op 80x60)
  float cx = 0, cy = 0;
  int sx = 0, sy = 0;                   // gevonden eigen verschuiving
};

struct HeadInfo {             // van de RP2350
  float yaw = 0, elev = 0, yaw_rate = 0, gyro_z = 0;
  bool fix = true;
};

struct Vision {
  VisionCfg cfg;
#define VLAG_MAX 3
  uint8_t cur[VW * VH], prev[VW * VH], diff[VW * VH];
  uint8_t hist[VLAG_MAX][VW * VH]; int hist_n = 0, hist_i = 0;
  float E[CW * CH];
  bool have_prev = false;
  VisionOut out;
  float wb_prev = 0, d_prev = 1.0f, t_prev = 0;
  bool track = false;

  // licht veranderd (flitsled): oude beelden weggooien, anders lijkt het hele beeld te bewegen
  void relight() { hist_n = 0; hist_i = 0; track = false; }
  void reset() { have_prev = false; hist_n = 0; hist_i = 0; memset(E, 0, sizeof(E)); out = VisionOut(); track = false; }

  // 160x120 grijs -> 80x60 (gemiddelde van 2x2)
  // Beeld rechtop zetten (rot) en verkleinen naar 80x60. Bij 90/270 gr is het beeld staand: de korte kant
  // van de sensor wordt de breedte (80 px), de HELE lange kant gaat in de 60 px hoogte (2026-11-03: niet meer
  // bijsnijden - de verwerkte pixels zijn dan niet vierkant, de hoekberekening houdt daar rekening mee).
  bool portrait() const { return cfg.rot == 90 || cfg.rot == 270; }
  void load(const uint8_t* img, int w, int h) {
    float scx = portrait() ? (float)h / VW : (float)w / VW;         // sensor-px per verwerkte px (breedte)
    float scy = portrait() ? (float)w / VH : (float)h / VH;         // idem (hoogte)
    for (int y = 0; y < VH; y++)
      for (int x = 0; x < VW; x++) {
        float X = x * scx, Y = y * scy;                             // in het rechtop gezette beeld
        int sx, sy;
        switch (cfg.rot) {
          case 90:  sx = (int)Y;           sy = (int)(h - 1 - X); break;   // met de klok mee gedraaid
          case 180: sx = (int)(w - 1 - X); sy = (int)(h - 1 - Y); break;
          case 270: sx = (int)(w - 1 - Y); sy = (int)X;           break;
          default:  sx = (int)X;           sy = (int)Y;
        }
        if (sx < 0) sx = 0; if (sx > w - 2) sx = w - 2;
        if (sy < 0) sy = 0; if (sy > h - 2) sy = h - 2;
        const uint8_t* p = img + sy * w + sx;
        int v = p[0] + p[1] + p[w] + p[w + 1];
        cur[y * VW + x] = (uint8_t)(v >> 2);
      }
  }

  void find_shift(int* bx, int* by) {   // globale verschuiving: vorige(x - dx, y - dy) ~ nu(x, y)
    int R = cfg.shift_r; long best = -1; *bx = *by = 0;
    for (int dy = -R; dy <= R; dy++)
      for (int dx = -R; dx <= R; dx++) {
        long s = 0;
        for (int y = R; y < VH - R; y += 2)
          for (int x = R; x < VW - R; x += 2) {
            int a = cur[y * VW + x] - prev[(y - dy) * VW + (x - dx)];
            s += a < 0 ? -a : a;
          }
        if (best < 0 || s < best || (s == best && dx * dx + dy * dy < (*bx) * (*bx) + (*by) * (*by))) { best = s; *bx = dx; *by = dy; }
      }
  }

  // t in s. Geeft true als er een nieuw resultaat is.
  bool process(const uint8_t* img, int w, int h, float t, const HeadInfo& hd) {
    load(img, w, h);
    int L = cfg.lag < 1 ? 1 : (cfg.lag > VLAG_MAX ? VLAG_MAX : cfg.lag);
    if (hist_n < L) { memcpy(hist[hist_i], cur, sizeof(cur)); hist_i = (hist_i + 1) % VLAG_MAX; hist_n++; t_prev = t; return false; }
    // referentie = het beeld van L beelden terug
    memcpy(prev, hist[(hist_i - L + VLAG_MAX) % VLAG_MAX], sizeof(prev));
    memcpy(hist[hist_i], cur, sizeof(cur)); hist_i = (hist_i + 1) % VLAG_MAX;
    have_prev = true;
    float dt = t - t_prev; if (dt <= 0) dt = 0.04f; t_prev = t;
    int sx, sy; find_shift(&sx, &sy); out.sx = sx; out.sy = sy;
    // verschilbeeld, cel-energie
    float cell[CW * CH]; memset(cell, 0, sizeof(cell));
    int thr = (int)cfg.diff_thr;
    for (int y = 0; y < VH; y++)
      for (int x = 0; x < VW; x++) {
        int px = x - sx, py = y - sy; int d = 0;
        if (px >= 0 && px < VW && py >= 0 && py < VH) {
          d = cur[y * VW + x] - prev[py * VW + px]; d = d < 0 ? -d : d;
          if (d < thr) d = 0;
        }
        diff[y * VW + x] = (uint8_t)(d > 255 ? 255 : d);
        cell[(y / CS) * CW + x / CS] += d;
      }
    float a = cfg.smooth;
    // snelle kopbeweging: het beeld is onbetrouwbaar -> energie laten uitdoven, niets nieuws
    bool saccade = fabsf(hd.yaw_rate) > 1.5f;
    for (int i = 0; i < CW * CH; i++) E[i] = (1 - a) * E[i] + (saccade ? 0.0f : a * cell[i] / (CS * CS));
    // sterkste cel
    int pk = 0; for (int i = 1; i < CW * CH; i++) if (E[i] > E[pk]) pk = i;
    float peak = E[pk];
    VisionOut o = out;
    o.vis = peak >= cfg.min_energy;
    if (o.vis) {
      // aaneengesloten cellen boven 25% van de piek (flood fill)
      static uint8_t mark[CW * CH]; memset(mark, 0, sizeof(mark));
      int stack[CW * CH], sp = 0; stack[sp++] = pk; mark[pk] = 1;
      float lim = 0.25f * peak, sw = 0, swx = 0, swy = 0; int n = 0, cx0 = CW, cy0 = CH, cx1 = -1, cy1 = -1;
      while (sp) {
        int c = stack[--sp]; int x = c % CW, y = c / CW;
        float e = E[c]; n++; sw += e; swx += e * (x + 0.5f); swy += e * (y + 0.5f);
        if (x < cx0) { cx0 = x; } if (x > cx1) { cx1 = x; } if (y < cy0) { cy0 = y; } if (y > cy1) { cy1 = y; }
        const int nb[4] = {c - 1, c + 1, c - CW, c + CW};
        const bool ok[4] = {x > 0, x < CW - 1, y > 0, y < CH - 1};
        for (int k = 0; k < 4; k++) if (ok[k] && !mark[nb[k]] && E[nb[k]] > lim) { mark[nb[k]] = 1; stack[sp++] = nb[k]; }
      }
      o.cx = swx / sw * CS; o.cy = swy / sw * CS;
      o.x0 = cx0 * CS; o.y0 = cy0 * CS; o.x1 = (cx1 + 1) * CS; o.y1 = (cy1 + 1) * CS;
      o.area = (float)n / (CW * CH);
      o.inten = sw / n;
      // richting (pinhole), + = links / omhoog
      // fov_h/fov_v = van de sensor (lange/korte kant). Staand: breedte = korte kant, hoogte = deel van de lange
      float fh = portrait() ? cfg.fov_v_deg : cfg.fov_h_deg;
      float fx = (VW / 2.0f) / tanf(fh * 0.5f * 0.0174533f);
      float fv = portrait() ? cfg.fov_h_deg : cfg.fov_v_deg;     // hele lange kant in de hoogte
      float fy = (VH / 2.0f) / tanf(fv * 0.5f * 0.0174533f);
      float u = o.cx - VW / 2.0f, v = o.cy - VH / 2.0f;
      if (cfg.flip_x) { u = -u; } if (cfg.flip_y) { v = -v; }
      o.bx = -atanf(u / fx); o.by = -atanf(v / fy);
      // afstand: onderkant van de vlek op de vloer (als die in beeld is), anders uit het oppervlak
      float vb = (cfg.flip_y ? (VH - o.y0) : o.y1) - VH / 2.0f;
      float below = atanf(vb / fy) - hd.elev;        // hoek onder de horizon
      float d;
      bool bottom_in = cfg.flip_y ? (o.y0 > 1) : (o.y1 < VH - 1);
      if (bottom_in && below > 0.07f) d = cfg.cam_h / tanf(below);
      else d = cfg.size_k / sqrtf(o.area > 1e-3f ? o.area : 1e-3f);
      d = d < 0.12f ? 0.12f : (d > 3.0f ? 3.0f : d);
      o.dist = track ? out.dist + fminf(1.0f, dt / 0.4f) * (d - out.dist) : d;
      // niveaus met een beetje hysterese
      auto level = [](float x, float t1, float t2, float t3, int prev) {
        float h = 0.85f;   // naar beneden pas bij 85% van de drempel
        int l = x >= t3 ? 3 : x >= t2 ? 2 : x >= t1 ? 1 : 0;
        if (l < prev) { int lh = x >= t3 * h ? 3 : x >= t2 * h ? 2 : x >= t1 * h ? 1 : 0; if (lh > l) l = lh; }
        return l;
      };
      o.mint = level(o.inten, cfg.i1, cfg.i2, cfg.i3, out.mint);
      o.msize = level(o.area, cfg.s1 > 0 ? cfg.s1 : 1e-9f, cfg.s2, cfg.s3, out.msize);
      if (o.msize < 1) o.msize = 1;
      if (o.mint < 1) o.mint = 1;
      // eigen beweging t.o.v. de wereld en nadering, ~0.8 s gemiddeld (= training)
      float wb = o.bx + hd.yaw;
      float k = fminf(1.0f, dt / 0.8f);
      if (track) {
        float rate = (wb - wb_prev) / dt + hd.gyro_z;
        o.tflow = out.tflow + k * (rate - out.tflow);
        o.approach = out.approach + k * (-(o.dist - d_prev) / dt - out.approach);
      } else { o.tflow = 0; o.approach = 0; }
      wb_prev = wb; d_prev = o.dist; track = true;
    } else {
      o.msize = o.mint = 0; o.area = o.inten = 0; track = false;
      o.tflow *= 0.9f; o.approach *= 0.9f;
    }
    out = o;
    return true;
  }
};
