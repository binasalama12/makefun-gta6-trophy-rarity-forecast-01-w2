# gta6-trophy-rarity-forecast-01 — wallet w2

Forecast each of 55 GTA VI PS5 trophies' Exophase earn rate (%) at the
case target, as five quantiles (5/25/50/75/95%) per trophy.

## Approach — two-component counts model

An earn rate is a ratio, `earned / players_tracked`, and its two counts
move for different reasons: new players dilute every rate (denominator
grows) while players earning trophies push individual rates up
(numerator grows). So the program fits each count separately as a
log-linear trend in time and divides at the target:

- `players_tracked`: least-squares fit of log(players) vs hours over the
  last 24h of reads; slope floored at 0, capped at 0.35 log units/hour.
- Each trophy's earned count (`rate/100 * players_tracked` per read):
  least-squares fit of log(earned) vs hours; slope floored at 0, capped
  at 0.5/hour; forecast earned is capped at forecast players.
- Median rate = `100 * earned_target / players_target`, clamped to
  [0.01, 99.99]. A trophy at 100% keeps earned == players, so it stays
  at 100%.

**Flat history / dilution branch.** On launch night Exophase did not
refresh for hours (every read: 638 players, identical rates), yet the
practice answers show the next refresh jumping to 3,868 players with
every rate lower. When history is flat (one distinct players value, no
rate moved), the program assumes a refresh is coming: players multiply
by ~3x over a 90-minute horizon, scaling as `3 ** (horizon/90min)`
(capped at 50x), earned counts frozen — so every rate is diluted by the
same factor while the 100% trophy stays put.

**Spread.** Quantiles from a normal on the logit scale around the
median: `sigma = 0.5*sqrt(horizon_days) + 0.35 + 0.8 if flat`. The
0.5-term is the random-walk part, 0.35 model uncertainty in the two
fitted trends, and the extra 0.8 in the flat branch is refresh
uncertainty (unknown size/timing of the next Exophase refresh).
Quantiles use z = -1.645, -0.674, 0, 0.674, 1.645, are rounded to 4
decimals (half up) and forced non-decreasing.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in`
in `$MAKEFUN_INPUT_DIR` (else `/input`), writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`). Deterministic: no randomness,
files processed in name order.

## Files

- `main.py` — the whole entry (single file, no dependencies).
- `README.md` — this file.
- `LICENSE` — MIT.
