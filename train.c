#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ai.h"

/* Headless pong for evolutionary training.
 * Field mirrors the SDL game (800x600, paddle 14x100, ball 12, speeds). */

#define WIN_W 800
#define WIN_H 600
#define PADDLE_W 14
#define PADDLE_H 100
#define BALL_SZ 12
#define AI_SPEED 6
#define BALL_SPEED 6
#define MAX_FRAMES 4000
#define WIN_SCORE 5

#define POP 40
#define ELITE 8
#define GENERATIONS 300
#define MATCHES_PER_GENOME 4
#define MUTATION_RATE 0.15f
#define MUTATION_SIGMA 0.3f

typedef struct {
    float w[NN_NPARAMS];
    float fitness;
} Genome;

static float frand(void) { return (float)rand() / (float)RAND_MAX; }
static float frand_sym(void) { return frand()*2.0f - 1.0f; }

static float gauss(void) {
    float u1 = frand() + 1e-9f;
    float u2 = frand();
    return sqrtf(-2.0f*logf(u1)) * cosf(6.2831853f*u2);
}

/* Mirror state for left paddle so same network can play both sides. */
static void mirror_state(AIState *m, const AIState *s) {
    *m = *s;
    m->ball_x = s->field_w - s->ball_x;
    m->ball_dx = -s->ball_dx;
    m->paddle_x = s->field_w - s->paddle_x;
}

static int rect_overlap(int ax, int ay, int aw, int ah,
                        int bx, int by, int bw, int bh) {
    return !(ax + aw <= bx || bx + bw <= ax ||
             ay + ah <= by || by + bh <= ay);
}

/* Returns point differential for left player (L - R). */
static int play_match(const float *wL, const float *wR) {
    int lx = 20, ly = WIN_H/2 - PADDLE_H/2;
    int rx = WIN_W - 20 - PADDLE_W, ry = WIN_H/2 - PADDLE_H/2;
    int bx = WIN_W/2, by = WIN_H/2;
    int bdx = (rand()%2 ? 1 : -1) * BALL_SPEED;
    int bdy = (rand()%2 ? 1 : -1) * (BALL_SPEED - 2);
    int scoreL = 0, scoreR = 0;
    int hits = 0;

    for (int f = 0; f < MAX_FRAMES; f++) {
        AIState sR = { by, bx, bdy, bdx, ry, PADDLE_H, WIN_H, WIN_W, rx };
        AIState sL; mirror_state(&sL, &sR); sL.paddle_y = ly;

        int mvR = ai_forward(wR, &sR);
        int mvL = ai_forward(wL, &sL);

        ly += mvL * AI_SPEED;
        ry += mvR * AI_SPEED;
        if (ly < 0) ly = 0;
        if (ly + PADDLE_H > WIN_H) ly = WIN_H - PADDLE_H;
        if (ry < 0) ry = 0;
        if (ry + PADDLE_H > WIN_H) ry = WIN_H - PADDLE_H;

        bx += bdx;
        by += bdy;
        if (by <= 0) { by = 0; bdy = -bdy; }
        if (by + BALL_SZ >= WIN_H) { by = WIN_H - BALL_SZ; bdy = -bdy; }

        if (bdx < 0 && rect_overlap(bx, by, BALL_SZ, BALL_SZ, lx, ly, PADDLE_W, PADDLE_H)) {
            bx = lx + PADDLE_W; bdx = -bdx;
            int hit = (by + BALL_SZ/2) - (ly + PADDLE_H/2);
            bdy = hit / 8; if (bdy == 0) bdy = (rand()%2 ? 1 : -1);
            hits++;
        }
        if (bdx > 0 && rect_overlap(bx, by, BALL_SZ, BALL_SZ, rx, ry, PADDLE_W, PADDLE_H)) {
            bx = rx - BALL_SZ; bdx = -bdx;
            int hit = (by + BALL_SZ/2) - (ry + PADDLE_H/2);
            bdy = hit / 8; if (bdy == 0) bdy = (rand()%2 ? 1 : -1);
            hits++;
        }

        if (bx + BALL_SZ < 0) {
            scoreR++;
            bx = WIN_W/2; by = WIN_H/2;
            bdx = BALL_SPEED; bdy = (rand()%2 ? 1 : -1)*(BALL_SPEED-2);
        }
        if (bx > WIN_W) {
            scoreL++;
            bx = WIN_W/2; by = WIN_H/2;
            bdx = -BALL_SPEED; bdy = (rand()%2 ? 1 : -1)*(BALL_SPEED-2);
        }
        if (scoreL >= WIN_SCORE || scoreR >= WIN_SCORE) break;
    }

    /* Reward hits a bit to bootstrap learning from random init. */
    return (scoreL - scoreR) * 10 + hits;
}

static void randomize(Genome *g) {
    for (int i = 0; i < NN_NPARAMS; i++) g->w[i] = frand_sym() * 0.5f;
    g->fitness = 0;
}

static void mutate(Genome *g) {
    for (int i = 0; i < NN_NPARAMS; i++) {
        if (frand() < MUTATION_RATE) g->w[i] += gauss() * MUTATION_SIGMA;
    }
}

static void crossover(Genome *child, const Genome *a, const Genome *b) {
    for (int i = 0; i < NN_NPARAMS; i++) {
        child->w[i] = (frand() < 0.5f) ? a->w[i] : b->w[i];
    }
    child->fitness = 0;
}

static int cmp_fitness_desc(const void *a, const void *b) {
    float fa = ((const Genome*)a)->fitness;
    float fb = ((const Genome*)b)->fitness;
    return (fa < fb) - (fa > fb);
}

int main(int argc, char **argv) {
    unsigned seed = (unsigned)time(NULL);
    int warm_start = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--resume")) warm_start = 1;
    }
    srand(seed);
    printf("[train] seed=%u pop=%d gens=%d params=%d\n",
           seed, POP, GENERATIONS, NN_NPARAMS);

    Genome *pop = calloc(POP, sizeof(Genome));
    for (int i = 0; i < POP; i++) randomize(&pop[i]);

    if (warm_start && ai_load(AI_WEIGHTS_FILE) == 0) {
        ai_get_weights(pop[0].w);
        for (int i = 1; i < POP/2; i++) {
            memcpy(pop[i].w, pop[0].w, sizeof pop[0].w);
            mutate(&pop[i]);
        }
        printf("[train] resumed from %s\n", AI_WEIGHTS_FILE);
    }

    for (int gen = 0; gen < GENERATIONS; gen++) {
        for (int i = 0; i < POP; i++) pop[i].fitness = 0;

        for (int i = 0; i < POP; i++) {
            for (int m = 0; m < MATCHES_PER_GENOME; m++) {
                int j = rand() % POP;
                if (j == i) j = (j+1) % POP;
                int diff = play_match(pop[i].w, pop[j].w);
                pop[i].fitness += diff;
                pop[j].fitness -= diff;
            }
        }

        qsort(pop, POP, sizeof(Genome), cmp_fitness_desc);
        printf("[gen %3d] best=%.1f  median=%.1f  worst=%.1f\n",
               gen, pop[0].fitness, pop[POP/2].fitness, pop[POP-1].fitness);

        ai_set_weights(pop[0].w);
        ai_save(AI_WEIGHTS_FILE);

        Genome *next = calloc(POP, sizeof(Genome));
        for (int i = 0; i < ELITE; i++) next[i] = pop[i];
        for (int i = ELITE; i < POP; i++) {
            int a = rand() % ELITE;
            int b = rand() % ELITE;
            crossover(&next[i], &pop[a], &pop[b]);
            mutate(&next[i]);
        }
        memcpy(pop, next, sizeof(Genome)*POP);
        free(next);
    }

    ai_set_weights(pop[0].w);
    ai_save(AI_WEIGHTS_FILE);
    printf("[train] best saved to %s\n", AI_WEIGHTS_FILE);
    free(pop);
    return 0;
}
