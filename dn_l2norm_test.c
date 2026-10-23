/* Independent arithmetic witnesses, not a second copy of the GPU reduction. */
#include "deltanet.h"
#include <math.h>
#include <stdio.h>
int main(void) {
    const float scales[] = {0.f, 1e-8f, 1e-3f, 1.f, 100.f};
    for (unsigned s = 0; s < sizeof scales / sizeof scales[0]; s++) {
        float q[128] = {0}, out[128]; q[0] = scales[s];
        l2norm128_gpu(q, 1e-6f, out);
        double expected = (double)q[0] / sqrt((double)q[0] * q[0] + (double)1e-6f);
        if (!isfinite(out[0]) || fabs((double)out[0] - expected) > 2e-7) {
            fprintf(stderr, "x=%g: got %.9g expected %.9g\n", q[0], out[0], expected); return 1;
        }
        for (int k = 1; k < 128; k++) if (out[k] != 0.f) return 1;
        printf("x=%g q[0]=%.9g expected=%.9g\n", q[0], out[0], expected);
    }
    puts("PASS: x / sqrt(sum(x*x) + eps), including zero and tiny inputs");
    return 0;
}
