#include "ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Layout of flat weight vector:
 *   [0 .. IN*HID)                     W1  (IN x HID)
 *   [IN*HID .. IN*HID+HID)            b1
 *   [... .. + HID*OUT)                W2  (HID x OUT)
 *   [... .. + OUT)                    b2
 */

static float g_w[NN_NPARAMS];
static int   g_loaded = 0;

static float frand_sym(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void normalize(const AIState *s, float *x) {
    float fh = (float)s->field_h;
    float fw = (float)s->field_w;
    x[0] = (float)s->ball_y / fh;
    x[1] = (float)s->ball_x / fw;
    x[2] = (float)s->ball_dy / 10.0f;
    x[3] = (float)s->ball_dx / 10.0f;
    x[4] = (float)s->paddle_y / fh;
}

int ai_forward(const float *w, const AIState *s) {
    float x[NN_IN];
    normalize(s, x);

    const float *W1 = w;
    const float *b1 = w + NN_IN*NN_HID;
    const float *W2 = b1 + NN_HID;
    const float *b2 = W2 + NN_HID*NN_OUT;

    float h[NN_HID];
    for (int j = 0; j < NN_HID; j++) {
        float s1 = b1[j];
        for (int i = 0; i < NN_IN; i++) s1 += x[i] * W1[i*NN_HID + j];
        h[j] = tanhf(s1);
    }

    float o[NN_OUT];
    int best = 0;
    for (int k = 0; k < NN_OUT; k++) {
        float s2 = b2[k];
        for (int j = 0; j < NN_HID; j++) s2 += h[j] * W2[j*NN_OUT + k];
        o[k] = s2;
        if (s2 > o[best]) best = k;
    }
    return best - 1; /* 0,1,2 -> -1,0,1 */
}

void ai_set_weights(const float *w) {
    memcpy(g_w, w, sizeof(float)*NN_NPARAMS);
    g_loaded = 1;
}

void ai_get_weights(float *w) {
    memcpy(w, g_w, sizeof(float)*NN_NPARAMS);
}

void ai_random_weights(unsigned seed) {
    srand(seed);
    float scale = 0.5f;
    for (int i = 0; i < NN_NPARAMS; i++) g_w[i] = frand_sym() * scale;
    g_loaded = 1;
}

int ai_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(g_w, sizeof(float), NN_NPARAMS, f);
    fclose(f);
    if (n != NN_NPARAMS) return -1;
    g_loaded = 1;
    return 0;
}

int ai_save(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t n = fwrite(g_w, sizeof(float), NN_NPARAMS, f);
    fclose(f);
    return (n == NN_NPARAMS) ? 0 : -1;
}

int ai_move(const AIState *s) {
    if (!g_loaded) {
        if (ai_load(AI_WEIGHTS_FILE) != 0) {
            fprintf(stderr, "[ai] %s not found, using random weights. Run ./train first.\n",
                    AI_WEIGHTS_FILE);
            ai_random_weights(42);
        }
    }
    return ai_forward(g_w, s);
}
