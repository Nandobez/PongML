<div align="center">

<pre>
██████╗  ██████╗ ███╗   ██╗ ██████╗ ███╗   ███╗██╗
██╔══██╗██╔═══██╗████╗  ██║██╔════╝ ████╗ ████║██║
██████╔╝██║   ██║██╔██╗ ██║██║  ███╗██╔████╔██║██║
██╔═══╝ ██║   ██║██║╚██╗██║██║   ██║██║╚██╔╝██║██║
██║     ╚██████╔╝██║ ╚████║╚██████╔╝██║ ╚═╝ ██║███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
</pre>

### Pong with a Neuro-Evolution AI — in pure C

</div>

A tiny SDL Pong where the opponent is a feed-forward neural network trained from scratch with a genetic algorithm. No Python, no libraries beyond SDL2 and libm.

## Architecture

```
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│   train.c    │  GA    │  weights.bin │  load  │   pong.c     │
│  (headless)  │ ─────▶ │  (300 bytes) │ ─────▶ │  (SDL game)  │
└──────────────┘        └──────────────┘        └──────────────┘
```

- **`ai.c` / `ai.h`** — fully-connected network: **5 inputs → 8 hidden → 3 outputs** (`up / stay / down`). 75 float weights total, persisted as `weights.bin`.
- **`train.c`** — headless simulator + genetic algorithm trainer.
- **`pong.c`** — SDL2 game; player vs AI.

## Network inputs

| # | Feature       |
|---|---------------|
| 1 | `ball_y`      |
| 2 | `ball_x`      |
| 3 | `ball_dy`     |
| 4 | `ball_dx`     |
| 5 | `paddle_y`    |

Output: `argmax` of 3 logits → `-1 / 0 / +1` paddle delta.

## Training (genetic algorithm)

| Param                | Value |
|----------------------|-------|
| Population           | 40    |
| Elite                | 8     |
| Generations          | 300   |
| Matches / genome     | 4     |
| Mutation rate        | 0.15  |
| Mutation σ           | 0.3   |
| Win score            | 5     |
| Max frames / match   | 4000  |

Fitness = wins + tiebreakers (rallies, ball-x). Elites survive untouched; the rest is bred from tournament-selected parents with Gaussian mutation.

## Build & run

```bash
# Train (writes weights.bin)
cc train.c ai.c -o train -lm
./train

# Play (loads weights.bin)
cc pong.c ai.c -o pong $(pkg-config --cflags --libs sdl2) -lm
./pong
```

## Files

```
Pong/
├── ai.h          # NN definition (NN_IN=5, NN_HID=8, NN_OUT=3)
├── ai.c          # Forward pass + weight I/O
├── train.c       # Headless GA trainer
├── pong.c        # SDL2 game
└── weights.bin   # Trained weights (300 B)
```

## License

MIT.
