// test_vision.cpp - beeldverwerking van de camera op de pc testen met nagemaakte beelden.
//   g++ -O2 -I firmware/duckling_cam tools/test_vision.cpp -o test_vision && ./test_vision
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vision.h"

static uint8_t img[160 * 120];
static float bg[400 * 300];                       // grote achtergrond (structuur), voor kopdraaien
static void make_bg() { srand(1); for (int i = 0; i < 400 * 300; i++) bg[i] = 60 + 80 * ((i / 7 + (i / 400) / 9) % 5 == 0) + 30 * sinf(i * 0.013f); }
// pan_px: kop gedraaid (beeld schuift), obj: vierkant op (ox, oy) grootte s, helderheid +b
static void render(float pan_px, float ox, float oy, float s, float b, float noise) {
  for (int y = 0; y < 120; y++)
    for (int x = 0; x < 160; x++) {
      int X = (int)(x + 120 + pan_px), Y = y + 90;
      float v = bg[Y * 400 + X];
      if (fabsf(x - ox) < s / 2 && fabsf(y - oy) < s / 2) v += b;
      v += noise * ((rand() % 1000) / 500.0f - 1.0f);
      img[y * 160 + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}
static void scenario(const char* name, int frames, float (*f)(int, float*, float*, float*, float*, float*)) {
  Vision v; v.reset(); HeadInfo hd; float t = 0;
  int vis = 0; float sbx = 0, sarea = 0; int last_ms = 0, last_mi = 0; float tf = 0, ap = 0, dist = 0;
  for (int k = 0; k < frames; k++) {
    float pan, ox, oy, s, b; float yaw_rate = f(k, &pan, &ox, &oy, &s, &b);
    hd.yaw = -pan / 160.0f * 1.047f;   // pan in px -> kophoek (+ = links: beeld schuift naar rechts)
    hd.yaw_rate = yaw_rate;
    render(pan, ox, oy, s, b, 4.0f);
    t += 0.04f;
    v.process(img, 160, 120, t, hd);
    if (k > frames / 2) { vis += v.out.vis; if (v.out.vis) { sbx += v.out.bx; sarea += v.out.area; } last_ms = v.out.msize; last_mi = v.out.mint; tf += v.out.tflow / (frames - frames / 2 - 1); ap += v.out.approach / (frames - frames / 2 - 1); dist = v.out.dist; }
  }
  int n = frames - frames / 2 - 1;
  printf("%-34s zichtbaar %3d%%  bx %+5.1f gr  oppervlak %.3f  grootte %d  intensiteit %d  tflow(gem) %+.2f rad/s  nadering(gem) %+.2f  afstand %.2f\n",
         name, 100 * vis / n, vis ? sbx / vis * 57.3f : 0.0f, vis ? sarea / vis : 0.0f, last_ms, last_mi, tf, ap, dist);
}
// rot_test: zelfde hand links, maar de camera 90 gr gedraaid gemonteerd -> na rot=90 moet hij links blijven
static void rot_test(int rot) {
  Vision v; v.reset(); v.cfg.rot = rot; HeadInfo hd; float t = 0; int n = 0; float sbx = 0, sby = 0;
  static uint8_t r[160 * 120];
  for (int k = 0; k < 100; k++) {
    render(0, 40 + 8 * sinf(k * 0.9f), 40, 14, 70, 4.0f);     // rechtop: hand links-boven
    // wat een gedraaide sensor ziet: rechtop-beeld (160x120) terugdraaien; we simuleren alleen 180
    for (int y = 0; y < 120; y++) for (int x = 0; x < 160; x++) r[y * 160 + x] = rot == 180 ? img[(119 - y) * 160 + (159 - x)] : img[y * 160 + x];
    t += 0.04f; v.process(r, 160, 120, t, hd);
    if (k > 50 && v.out.vis) { n++; sbx += v.out.bx; sby += v.out.by; }
  }
  printf("rot %3d: hand links-boven -> richting %+5.1f / %+5.1f gr (verwacht +,+)\n", rot, n ? sbx / n * 57.3f : 0, n ? sby / n * 57.3f : 0);
}
// staand gemonteerd: rechtop-beeld U is 120 breed x 160 hoog; hand links-boven. De sensor ziet U gedraaid.
static void rot_test_portrait(int rot) {
  Vision v; v.reset(); v.cfg.rot = rot; HeadInfo hd; float t = 0; int n = 0; float sbx = 0, sby = 0;
  static uint8_t U[120 * 160], S[160 * 120];
  for (int k = 0; k < 100; k++) {
    float hx = 30 + 6 * sinf(k * 0.9f), hy = 60;
    for (int Y = 0; Y < 160; Y++) for (int X = 0; X < 120; X++) {
      float b = bg[(Y + 50) * 400 + X + 100] + ((fabsf(X - hx) < 7 && fabsf(Y - hy) < 7) ? 70 : 0) + 4 * ((rand() % 1000) / 500.0f - 1);
      U[Y * 120 + X] = (uint8_t)(b < 0 ? 0 : b > 255 ? 255 : b);
    }
    for (int sy = 0; sy < 120; sy++) for (int sx = 0; sx < 160; sx++)
      S[sy * 160 + sx] = rot == 90 ? U[sx * 120 + (119 - sy)] : U[(159 - sx) * 120 + sy];   // = omgekeerde van load()
    t += 0.04f; v.process(S, 160, 120, t, hd);
    if (k > 50 && v.out.vis) { n++; sbx += v.out.bx; sby += v.out.by; }
  }
  printf("rot %3d (staand): hand links-boven -> richting %+5.1f / %+5.1f gr (verwacht +,+)\n", rot, n ? sbx / n * 57.3f : 0, n ? sby / n * 57.3f : 0);
}
int main() {
  make_bg();
  rot_test(0); rot_test(180); rot_test_portrait(90); rot_test_portrait(270);
  scenario("stil, niets beweegt", 100, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 0; *x = 80; *y = 60; *s = 0; *b = 0; return 0.0f; });
  scenario("kop draait (pan), niets beweegt", 100, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 30 * sinf(k * 0.08f); *x = 80; *y = 60; *s = 0; *b = 0; return 0.3f * cosf(k * 0.08f); });
  scenario("hand zwaait links (klein)", 100, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 0; *x = 40 + 8 * sinf(k * 0.9f); *y = 50; *s = 14; *b = 70; return 0.0f; });
  scenario("hand zwaait rechts (klein)", 100, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 0; *x = 120 + 8 * sinf(k * 0.9f); *y = 50; *s = 14; *b = 70; return 0.0f; });
  scenario("groot lijf beweegt midden", 100, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 0; *x = 80 + 12 * sinf(k * 0.5f); *y = 60; *s = 60; *b = 60; return 0.0f; });
  scenario("hand zwaait + kop draait", 100, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 20 * sinf(k * 0.05f); *x = 60 + 8 * sinf(k * 0.9f); *y = 50; *s = 14; *b = 70; return 0.15f * cosf(k * 0.05f); });
  scenario("loopt naar links (verwacht tflow ~+0.13)", 120, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 0; *x = 140 - k * 0.8f; *y = 60; *s = 20; *b = 60; return 0.0f; });
  scenario("komt dichterbij (groeit, nadering +)", 120, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 0; *x = 80 + 3 * sinf(k * 0.7f); *y = 60; *s = 8 + k * 0.4f; *b = 60; return 0.0f; });
  scenario("zwak, net onder de drempel", 100, [](int k, float* p, float* x, float* y, float* s, float* b) { *p = 0; *x = 80 + 2 * sinf(k * 0.9f); *y = 60; *s = 6; *b = 14; return 0.0f; });
}
