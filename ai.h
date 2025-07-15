#ifndef AI_H
#define AI_H

#define NN_IN 5
#define NN_HID 8
#define NN_OUT 3
#define NN_NPARAMS (NN_IN*NN_HID + NN_HID + NN_HID*NN_OUT + NN_OUT)
#define AI_WEIGHTS_FILE "weights.bin"

typedef struct {
    int ball_y;
    int ball_x;
    int ball_dy;
    int ball_dx;
    int paddle_y;
    int paddle_h;
    int field_h;
    int field_w;
    int paddle_x;
} AIState;

/* Main contract: -1 up, 0 stay, 1 down. Lazy-loads weights.bin on first call. */
int ai_move(const AIState *s);

/* Weight management (used by trainer). */
int  ai_load(const char *path);
int  ai_save(const char *path);
void ai_set_weights(const float *w);
void ai_get_weights(float *w);
void ai_random_weights(unsigned seed);

/* Forward pass on explicit weights (trainer uses this to avoid global state). */
int  ai_forward(const float *w, const AIState *s);

#endif
