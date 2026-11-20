"""gta6-trophy-rarity-forecast-01 -- wallet w2 entry.

Two-component counts model, written from scratch for this entry.

Idea: an Exophase earn rate is a ratio, earned / players_tracked, and the
two counts move for different reasons.  New players dilute every rate
(the denominator grows) while players actually earning trophies push
individual rates back up (the numerator grows).  So instead of fitting
the rate directly, this program fits each count separately as a
log-linear trend in time, forecasts both counts at the target, and
divides:

    rate_target = 100 * earned_target / players_target

History handling
---------------
* Reads are taken from the case's last 24 hours (all reads if fewer).
* players_tracked: least-squares fit of log(players) against hours
  since the first read.  Slope floored at 0 (the tracked population
  does not shrink) and capped at 0.35 log units/hour.
* Each trophy's earned count: earned = rate/100 * players_tracked per
  read.  Least-squares fit of log(max(earned, 0.5)) against time, slope
  floored at 0 (earnings do not un-happen) and capped at 0.5/hour.
  Forecast earned is capped at the forecast players, so no rate can
  exceed 100% through the counts.
* A trophy whose last rate is 100% has earned == players in every read,
  so its fitted earned slope equals the players slope and its forecast
  rate stays 100%.

Flat history / the dilution branch
----------------------------------
On launch night Exophase did not refresh for hours: every read identical
(638 players, same rates).  A flat fit would forecast "nothing changes",
but the practice answers show the refresh that follows: 3,868 players
and every rate lower.  So when history is flat (players_tracked takes a
single distinct value and no rate moved), the program assumes a refresh
is coming and the tracked population multiplies:

    players_target = players_last * 3 ** (horizon_minutes / 90)

i.e. ~3x over a 90-minute horizon, scaling with horizon length, capped
at a 50x multiple so a long flat horizon cannot explode.  Earned counts
are frozen in this branch (slope 0), so every rate is diluted by the
same factor -- the median of a 1.10% trophy over 90 minutes becomes
~0.37% -- while the 100% trophy stays at 100% because its earned count
grows with the population.

Spread
------
Quantiles come from a normal on the logit scale around the counts-model
median:

    sigma = 0.5 * sqrt(horizon_days) + 0.35 + (0.8 if flat else 0)

The 0.5*sqrt(days) term is the random-walk part, 0.35 is model
uncertainty in the two fitted trends, and the extra 0.8 in the flat
branch is the refresh uncertainty: the size and timing of the next
Exophase refresh are unknown, so the band must be wide enough to cover
both "no refresh yet" and "big refresh".

Contract: reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).  Python 3.13, standard
library only, no network, fully deterministic (no randomness, files in
name order, fixed float arithmetic).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Dilution prior for the flat branch: players multiply by this over 90 min.
DILUTION_BASE = 3.0
DILUTION_WINDOW_MIN = 90.0
DILUTION_MAX_MULT = 50.0

# Slope caps, in log units per hour.
PLAYERS_SLOPE_MAX = 0.35
EARNED_SLOPE_MAX = 0.5

# Spread parts, in logit units.
SIGMA_RW = 0.5          # random-walk coefficient per sqrt(day)
SIGMA_MODEL = 0.35      # uncertainty of the two fitted trends
SIGMA_FLAT_EXTRA = 0.8  # refresh uncertainty when history is flat


def parse_time(text):
    """ISO-8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def linfit(xs, ys):
    """Least-squares slope/intercept of ys on xs. Returns (slope, intercept)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, ys[0]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError("%s asks for levels %r, not %r" % (name, case["levels"], LEVELS))
    history = case["history"]
    trophies = case["trophies"]
    if not history:
        raise ValueError("%s has no history" % name)

    horizon_min = float(case["horizon_minutes"])
    horizon_days = horizon_min / 1440.0
    target = parse_time(case["target"])

    # Hours since the first read; window = last 24h of reads.
    t0 = parse_time(history[0]["at"])
    times = [(parse_time(h["at"]) - t0).total_seconds() / 3600.0 for h in history]
    t_target = (target - t0).total_seconds() / 3600.0
    cutoff = times[-1] - 24.0
    win = [i for i, t in enumerate(times) if t >= cutoff]
    if len(win) < 3:
        win = list(range(len(history)))
    wt = [times[i] for i in win]

    players = [float(h["players_tracked"]) for h in history]
    last_players = players[-1]

    # Flat detection: one distinct players value and no rate moved at all.
    flat_players = max(players) == min(players)
    rates_moved = False
    for h in history[1:]:
        for a, b in zip(h["rates"], history[0]["rates"]):
            if a is not None and b is not None and a != b:
                rates_moved = True
                break
        if rates_moved:
            break
    is_flat = flat_players and not rates_moved

    # --- Forecast players_tracked at the target. ---
    if is_flat:
        mult = DILUTION_BASE ** (horizon_min / DILUTION_WINDOW_MIN)
        mult = min(mult, DILUTION_MAX_MULT)
        players_target = last_players * mult
    else:
        wplayers = [players[i] for i in win]
        slope, intercept = linfit(wt, [math.log(max(p, 1.0)) for p in wplayers])
        slope = min(max(slope, 0.0), PLAYERS_SLOPE_MAX)
        players_target = math.exp(intercept + slope * t_target)
        # Never forecast fewer tracked players than the last read saw.
        players_target = max(players_target, last_players)

    # --- Spread on the logit scale. ---
    sigma = SIGMA_RW * math.sqrt(horizon_days) + SIGMA_MODEL
    if is_flat:
        sigma += SIGMA_FLAT_EXTRA

    rows = []
    for idx, trophy in enumerate(trophies):
        last_rate = history[-1]["rates"][idx]
        if last_rate is None:
            # Fall back to the most recent non-null read for this trophy.
            last_rate = 0.0
            for h in reversed(history):
                if h["rates"][idx] is not None:
                    last_rate = h["rates"][idx]
                    break

        if is_flat:
            if last_rate >= 99.995:
                # Everyone tracked has it; earned grows with the population.
                median = 99.99
            else:
                median = clamp_pct(last_rate / min(
                    DILUTION_BASE ** (horizon_min / DILUTION_WINDOW_MIN),
                    DILUTION_MAX_MULT))
        else:
            # Earned-count trend for this trophy over the window.
            xs = []
            ys = []
            for i in win:
                r = history[i]["rates"][idx]
                if r is None:
                    continue
                earned = r / 100.0 * players[i]
                xs.append(times[i])
                ys.append(math.log(max(earned, 0.5)))
            if len(xs) >= 2:
                slope, intercept = linfit(xs, ys)
            elif len(xs) == 1:
                slope, intercept = 0.0, ys[0]
            else:
                slope, intercept = 0.0, math.log(0.5)
            slope = min(max(slope, 0.0), EARNED_SLOPE_MAX)
            earned_target = math.exp(intercept + slope * t_target)
            # Earned can never exceed the tracked population.
            earned_target = min(earned_target, players_target * 0.9999)
            median = clamp_pct(100.0 * earned_target / players_target)

        lg = logit(median)
        qs = []
        for z in Z:
            q = clamp_pct(inv_logit(lg + z * sigma))
            qs.append(round4(q))
        # Rounding can tie or (never) invert neighbours; enforce order.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / ("%s.out" % path.stem)).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
