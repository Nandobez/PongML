#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ai.h"

#define WIN_W 800
#define WIN_H 600
#define PADDLE_W 14
#define PADDLE_H 100
#define BALL_SZ 12
#define PADDLE_SPEED 7
#define AI_SPEED 6
#define BALL_SPEED 6

typedef struct { int x, y; } Paddle;
typedef struct { int x, y, dx, dy; } Ball;

static void reset_ball(Ball *b, int dir) {
    b->x = WIN_W / 2 - BALL_SZ / 2;
    b->y = WIN_H / 2 - BALL_SZ / 2;
    b->dx = dir * BALL_SPEED;
    b->dy = ((rand() % 2) ? 1 : -1) * (BALL_SPEED - 2);
}

static void draw_digit(SDL_Renderer *r, int d, int x, int y, int s) {
    static const unsigned char seg[10] = {
        0x7E, 0x30, 0x6D, 0x79, 0x33, 0x5B, 0x5F, 0x70, 0x7F, 0x7B
    };
    /* bits: 0 top, 1 tr, 2 br, 3 bot, 4 bl, 5 tl, 6 mid */
    SDL_Rect segs[7] = {
        {x, y, 4*s, s},
        {x + 3*s, y, s, 3*s},
        {x + 3*s, y + 3*s, s, 3*s},
        {x, y + 6*s, 4*s, s},
        {x, y + 3*s, s, 3*s},
        {x, y, s, 3*s},
        {x, y + 3*s, 4*s, s},
    };
    for (int i = 0; i < 7; i++)
        if (seg[d] & (1 << (6 - i))) SDL_RenderFillRect(r, &segs[i]);
}

static void draw_score(SDL_Renderer *r, int n, int x, int y) {
    char buf[8];
    snprintf(buf, sizeof buf, "%d", n);
    int s = 6;
    for (int i = 0; buf[i]; i++) {
        draw_digit(r, buf[i] - '0', x + i * 6 * s, y, s);
    }
}

int main(void) {
    srand((unsigned)time(NULL));

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    if (ai_load(AI_WEIGHTS_FILE) != 0) {
        fprintf(stderr, "[pong] %s missing -- run ./train first (AI will play randomly)\n",
                AI_WEIGHTS_FILE);
        ai_random_weights(1);
    }

    SDL_Window *win = SDL_CreateWindow("Pong",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    Paddle player = { 20, WIN_H/2 - PADDLE_H/2 };
    Paddle ai = { WIN_W - 20 - PADDLE_W, WIN_H/2 - PADDLE_H/2 };
    Ball ball; reset_ball(&ball, 1);
    int score_p = 0, score_a = 0;
    int running = 1;

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = 0;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_q) running = 0;
        }

        const Uint8 *ks = SDL_GetKeyboardState(NULL);
        if (ks[SDL_SCANCODE_UP] || ks[SDL_SCANCODE_W]) player.y -= PADDLE_SPEED;
        if (ks[SDL_SCANCODE_DOWN] || ks[SDL_SCANCODE_S]) player.y += PADDLE_SPEED;
        if (player.y < 0) player.y = 0;
        if (player.y + PADDLE_H > WIN_H) player.y = WIN_H - PADDLE_H;

        AIState st = {
            .ball_y = ball.y, .ball_x = ball.x,
            .ball_dy = ball.dy, .ball_dx = ball.dx,
            .paddle_y = ai.y, .paddle_h = PADDLE_H,
            .field_h = WIN_H, .field_w = WIN_W,
            .paddle_x = ai.x
        };
        int mv = ai_move(&st);
        ai.y += mv * AI_SPEED;
        if (ai.y < 0) ai.y = 0;
        if (ai.y + PADDLE_H > WIN_H) ai.y = WIN_H - PADDLE_H;

        ball.x += ball.dx;
        ball.y += ball.dy;

        if (ball.y <= 0) { ball.y = 0; ball.dy = -ball.dy; }
        if (ball.y + BALL_SZ >= WIN_H) { ball.y = WIN_H - BALL_SZ; ball.dy = -ball.dy; }

        SDL_Rect br = { ball.x, ball.y, BALL_SZ, BALL_SZ };
        SDL_Rect pr = { player.x, player.y, PADDLE_W, PADDLE_H };
        SDL_Rect ar = { ai.x, ai.y, PADDLE_W, PADDLE_H };

        if (ball.dx < 0 && SDL_HasIntersection(&br, &pr)) {
            ball.x = player.x + PADDLE_W;
            ball.dx = -ball.dx;
            int hit = (ball.y + BALL_SZ/2) - (player.y + PADDLE_H/2);
            ball.dy = hit / 8;
            if (ball.dy == 0) ball.dy = (rand() % 2) ? 1 : -1;
        }
        if (ball.dx > 0 && SDL_HasIntersection(&br, &ar)) {
            ball.x = ai.x - BALL_SZ;
            ball.dx = -ball.dx;
            int hit = (ball.y + BALL_SZ/2) - (ai.y + PADDLE_H/2);
            ball.dy = hit / 8;
            if (ball.dy == 0) ball.dy = (rand() % 2) ? 1 : -1;
        }

        if (ball.x + BALL_SZ < 0) { score_a++; reset_ball(&ball, 1); }
        if (ball.x > WIN_W) { score_p++; reset_ball(&ball, -1); }

        SDL_SetRenderDrawColor(ren, 10, 10, 18, 255);
        SDL_RenderClear(ren);

        SDL_SetRenderDrawColor(ren, 60, 60, 80, 255);
        for (int y = 0; y < WIN_H; y += 20) {
            SDL_Rect d = { WIN_W/2 - 2, y, 4, 10 };
            SDL_RenderFillRect(ren, &d);
        }

        SDL_SetRenderDrawColor(ren, 230, 230, 240, 255);
        SDL_RenderFillRect(ren, &pr);
        SDL_RenderFillRect(ren, &ar);
        SDL_RenderFillRect(ren, &br);

        SDL_SetRenderDrawColor(ren, 180, 180, 200, 255);
        draw_score(ren, score_p, WIN_W/2 - 100, 30);
        draw_score(ren, score_a, WIN_W/2 + 60, 30);

        SDL_RenderPresent(ren);
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
