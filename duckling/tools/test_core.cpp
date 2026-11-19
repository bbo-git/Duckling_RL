// test_core.cpp - rekent duck_core.h op de pc door met dezelfde invoer als test_core.py (onnxruntime).
//   g++ -O2 -I firmware/duckling_loop tools/test_core.cpp -o test_core && ./test_core < invoer.txt > uit.txt
#include <stdio.h>
#include "duck_core.h"
static Policy pol; static Duck duck;
int main() {
  pol.init(); duck.reset();
  float g[3], gr[3], tw[3], hd[4], us[N_SERVO];
  while (scanf("%f %f %f %f %f %f %f %f %f %f %f %f %f", g, g + 1, g + 2, gr, gr + 1, gr + 2, tw, tw + 1, tw + 2,
               hd, hd + 1, hd + 2, hd + 3) == 13) {
    duck.step(pol, g, gr, tw, hd);
    joints_to_us(duck.filt, 0.0f, us);
    for (int j = 0; j < NJ; j++) printf("%.6f ", duck.act[j]);
    for (int j = 0; j < NJ; j++) printf("%.6f ", duck.filt[j]);
    for (int n = 0; n < N_SERVO; n++) printf("%.3f ", us[n]);
    printf("%.6f", duck.phase);
    for (int i = 0; i < N_OBS; i++) printf(" %.8g", duck.obs[i]);
    printf("\n");
  }
}
