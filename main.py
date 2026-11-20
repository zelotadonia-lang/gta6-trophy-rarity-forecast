"""Trophy Tide: a refresh-aware forecaster for GTA VI trophy earn rates.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at a case's target, from the history of
reads the case carries. Standard library only, no network, deterministic.

Model, per trophy
-----------------
An earn rate is earners / players-tracked, and Exophase only refreshes its
counts on its own schedule: between refreshes the page is frozen, at a
refresh the tracked count jumps and every rate whose earners did not keep up
falls. So the forecast is built from two views of the same history, then
blended:

1. Trend view: a recency-weighted least-squares line of logit(rate) against
   ln(1 + hours since release), fitted on the last day of reads only. It
   follows the drift of the rate itself.
2. Structural view: earners and players-tracked are fitted separately, both
   against ln(1 + hours since release) on the same recent window, and the
   rate at the target is earners(target) / tracked(target). Earners move
   smoothly (players keep earning), tracked jumps at refreshes, so splitting
   them keeps a refresh from bending the trend the way it bends the raw
   rate. Both fits use the recent window because a line fitted over the
   whole history is bent by the explosive first day, when every rate moved
   fastest, and extrapolated it throws a saturating rate to 0% or 100%.

The blended move from the last read is damped (a slope fitted over a short
history must not be trusted far ahead) and capped, so a long horizon cannot
throw a rate to 0% or 100% on a thin trend.

Spread
------
The quantiles come from a normal spread on the logit scale whose sigma mixes
the house random-walk term (0.5 * sqrt(days)), the standard error of the
trend at the target, and a refresh term: the spread of the logit jumps the
history's own refreshes caused, scaled by how many refreshes the horizon
should hold. Where the history holds no refresh at all (a flat page, as on
launch night), the forecast instead uses an honest two-scenario mixture: no
refresh before the target, or one refresh that dilutes every rate the way
the launch-night refresh did (about 1.8 logit units, the 638 -> 3,868
tracked jump the rules describe), with the refresh probability from a
five-hour prior interval. The quantiles are the mixture's, found by
bisection on its CDF.

Edge cases: a trophy at 100% stays there (every tracked player holds it,
new players get it at once); a trophy at 0% gets a low median with a wide
upward tail instead of a confident zero; a read that lacks a trophy is
skipped for that trophy; with no usable history a wide, low forecast is
written rather than crashing, because an invalid case forfeits the prize.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is one file; the line is
# kept so the layout stays the one the starter documents.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread of a rate's move, logit units per square-root day.
SIGMA_RW = 0.5
# Blend weight of the structural (earners / tracked) view vs the trend view.
W_STRUCTURAL = 0.65
# Recency half-life of a read in the trend fits, in hours.
HALF_LIFE_H = 24.0
# Window the trend is fitted on: only the recent past carries the slope
# that matters ahead. A line fitted over the whole history is bent by the
# explosive first day, when every rate moved fastest, and extrapolated it
# throws a saturating rate to 0% or 100%.
RECENT_H = 24.0
# Prior refresh interval (hours) and logit drop when no refresh is observed.
# Launch night: the page sat unchanged for hours, then refreshed with
# tracked 638 -> 3,868, a ~1.8 logit fall for a ~1% trophy.
PRIOR_REFRESH_INTERVAL_H = 5.0
PRIOR_REFRESH_DROP = 1.8
# Largest logit move allowed away from the last read: grows slowly with
# the horizon, so a long horizon cannot throw a rate across the scale.
# A move of 1.3 logit units is already a ~3.7x change in the odds.
MAX_MOVE_BASE = 0.6
MAX_MOVE_PER_SQRT_DAY = 0.35
MAX_MOVE_CAP = 1.5
# Guards on the structural view: earners never fall and tracked never
# falls, and neither is trusted to more than quadruple over the horizon.
GROWTH_CAP = 4.0
EARNERS_FLOOR = 0.9


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-max(-30.0, min(30.0, lg))))


def round4(x):
    """Round to 4 decimals, half up, as the starter does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (fraction kept)."""
    text = text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(text).timestamp()


def parse_release(game):
    """Release instant from the case's game block, midnight UTC if a date."""
    raw = (game or {}).get("release_date") or "2026-11-19"
    raw = str(raw).strip()
    if len(raw) == 10:
        raw = raw + "T00:00:00Z"
    return parse_time(raw)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (a, b, se_at) where se_at(x0) is the standard error of the
    fitted value at x0. With one point the slope is 0; with no spread in x
    the slope is 0 as well.
    """
    sw = sum(ws)
    if sw <= 0 or not xs:
        return 0.0, 0.0, 1.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid2 = sum(w * (y - my) ** 2 for w, y in zip(ws, ys))
        dof = max(1, len(xs) - 1)
        var = resid2 / sw / dof if sw else 0.0
        se = math.sqrt(max(var, 0.0))
        return my, 0.0, lambda x0, se=se: se
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(1, len(xs) - 2)
    var = resid2 / sw / dof if sw else 0.0

    def se_at(x0):
        lev = 1.0 / sw + (x0 - mx) ** 2 / sxx
        return math.sqrt(max(var * lev, 0.0))

    return a, b, se_at


def distinct_states(history):
    """Collapse consecutive identical reads, keeping the last time of each.

    A state is (players_tracked, rates tuple). Identical repeats carry no
    new information for a trend fit; keeping the last time marks when the
    state was most recently observed.
    """
    states = []
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if states and states[-1][0] == key:
            states[-1] = (key, parse_time(read["at"]))
        else:
            states.append((key, parse_time(read["at"])))
    return states


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    target_t = parse_time(case["target"])
    release_t = parse_release(case.get("game"))
    states = distinct_states(history)
    if not states:
        return {"trophies": [{"id": t["id"], "q": [round4(x) for x in (0.05, 0.3, 1.0, 3.0, 12.0)]} for t in trophies]}

    last_key, last_t = states[-1]
    last_tracked, last_rates = last_key
    hist_span_days = max((last_t - states[0][1]) / 86400.0, 0.0)

    def u_of(t):
        hours = (t - release_t) / 3600.0
        return math.log1p(max(hours, -0.99))

    u_target = u_of(target_t)
    u_last = u_of(last_t)

    # Recency weights per state, half-life a day, floored so old states
    # still count a little.
    weights = [max(0.05, 0.5 ** ((last_t - t) / (HALF_LIFE_H * 3600.0))) for _, t in states]
    u_vals = [u_of(t) for _, t in states]
    tracked_vals = [s[0][0] for s in states]

    # Recent window: the last day of states, or the last three states when
    # the history is shorter. Slopes are fitted here, never over the whole
    # history: the first day's explosive growth would bend any long fit.
    recent_idx = [k for k, (_, t) in enumerate(states) if last_t - t <= RECENT_H * 3600.0]
    if len(recent_idx) < 3:
        recent_idx = list(range(max(0, len(states) - 3), len(states)))
    recent_set = set(recent_idx)
    u_recent = [u_vals[k] for k in recent_idx]
    w_recent = [weights[k] for k in recent_idx]

    def recent_slope(us, ys, ws):
        """Slope of y ~ u on the recent window, from its last point."""
        if len(us) < 2:
            return 0.0
        a, b, _ = wls(us, ys, ws)
        return b

    # Global tracked fit on the recent window: log(tracked) ~ u, then the
    # tracked count at the target, guarded: tracked never falls, and it is
    # not trusted to more than quadruple over the horizon.
    t_pts = [(u_vals[k], math.log(max(tracked_vals[k] or 1, 1)), weights[k])
            for k in recent_idx if tracked_vals[k]]
    if len(t_pts) >= 2:
        b_t = recent_slope([p[0] for p in t_pts], [p[1] for p in t_pts], [p[2] for p in t_pts])
        tracked_target = (last_tracked or 1) * math.exp(b_t * (u_target - u_last))
    else:
        tracked_target = float(last_tracked or 1)
    tracked_target = min(max(tracked_target, float(last_tracked or 1)), GROWTH_CAP * float(last_tracked or 1))

    # Refreshes: states where tracked jumped >3% over the previous state.
    refresh_drops = []
    refresh_times = []
    for k in range(1, len(states)):
        prev_tracked = states[k - 1][0][0] or 0
        cur_tracked = states[k][0][0] or 0
        if prev_tracked and cur_tracked > prev_tracked * 1.03:
            diffs = []
            for i in range(len(trophies)):
                r_prev = states[k - 1][0][1][i] if i < len(states[k - 1][0][1]) else None
                r_cur = states[k][0][1][i] if i < len(states[k][0][1]) else None
                if r_prev is not None and r_cur is not None and r_prev < 99.99:
                    diffs.append(logit(r_cur) - logit(r_prev))
            if diffs:
                diffs.sort()
                refresh_drops.append(diffs[len(diffs) // 2])
                refresh_times.append(states[k][1])
    if refresh_drops:
        drops_sorted = sorted(refresh_drops)
        med_drop = drops_sorted[len(drops_sorted) // 2]
        refresh_drop = max(0.0, -med_drop)
        # Spread of the per-refresh median jumps, for the spread term.
        mean_drop = sum(refresh_drops) / len(refresh_drops)
        var_drop = sum((d - mean_drop) ** 2 for d in refresh_drops) / len(refresh_drops)
        refresh_sd = math.sqrt(var_drop)
        if len(refresh_times) >= 2 and hist_span_days > 0:
            interval_days = hist_span_days / len(refresh_times)
        else:
            interval_days = PRIOR_REFRESH_INTERVAL_H / 24.0
    else:
        refresh_drop = PRIOR_REFRESH_DROP
        refresh_sd = 1.0
        interval_days = PRIOR_REFRESH_INTERVAL_H / 24.0

    expected_refreshes = horizon_days / max(interval_days, 1e-6)
    p_refresh = 1.0 - math.exp(-expected_refreshes)
    # A refresh's jump mostly IS the trend over its interval, so only the
    # dispersion around the jumps' own median prices refresh noise, and it
    # is capped: staleness of one refresh interval is a small move.
    sigma_refresh = min(refresh_sd * math.sqrt(min(expected_refreshes, 9.0)), 0.8)

    # Damping for a far target: a slope fitted over the recent window is
    # trusted less the further ahead it is pushed.
    damp = 1.0 / (1.0 + 0.5 * (horizon_days / max(hist_span_days, 0.25)))
    max_move = min(MAX_MOVE_BASE + MAX_MOVE_PER_SQRT_DAY * math.sqrt(horizon_days), MAX_MOVE_CAP)

    rows = []
    for i, trophy in enumerate(trophies):
        pts = []
        for k, ((_, rates_k), t_k) in enumerate(states):
            rate = rates_k[i] if i < len(rates_k) else None
            if rate is None:
                continue
            pts.append((u_vals[k], rate, weights[k], tracked_vals[k], t_k, k))
        if not pts:
            rows.append({"id": trophy["id"], "q": [round4(x) for x in (0.05, 0.3, 1.0, 3.0, 12.0)]})
            continue
        last_rate = pts[-1][1]
        last_lg = logit(last_rate)

        # A trophy everyone holds stays held: new players earn it at once.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [round4(x) for x in (99.5, 99.9, 99.99, 99.99, 99.99)]})
            continue

        # A trophy nobody holds yet: no one has earned it in the history,
        # so the fits see a flat 0.00% and would swear it stays there. With
        # days of new players ahead, the honest forecast is a low median
        # with a wide upward tail, not a confident zero.
        if last_rate <= 0.0:
            rows.append({"id": trophy["id"], "q": [round4(x) for x in (0.01, 0.01, 0.03, 0.07, 0.21)]})
            continue

        # Trend view on logit(rate), fitted on the recent window only.
        r_pts = [p for p in pts if p[5] in recent_set] if len(pts) > 3 else pts
        a_y, b_y, se_y = wls([p[0] for p in r_pts], [logit(p[1]) for p in r_pts], [p[2] for p in r_pts])
        y_trend = a_y + b_y * u_target
        # The fit's own error, floored: a short window can look deceptively
        # exact, and the model itself is wrong by more than its residuals.
        se_trend = min(max(se_y(u_target), 0.15 + 0.10 * math.sqrt(horizon_days)), 1.2)

        # Structural view: log(earners) fitted on the recent window, then
        # earners(target) / tracked(target). Earners never fall and neither
        # side is trusted to more than quadruple over the horizon.
        e_pts = [(p[0], math.log(max(p[1] * p[3] / 100.0, 0.0) + 0.5), p[2])
                 for p in r_pts if p[3]]
        if len(e_pts) >= 2:
            b_e = recent_slope([p[0] for p in e_pts], [p[1] for p in e_pts], [p[2] for p in e_pts])
            earners_last = max(last_rate * (last_tracked or 1) / 100.0, 0.0) + 0.5
            earners_target = earners_last * math.exp(b_e * (u_target - u_last))
            earners_target = min(max(earners_target, EARNERS_FLOOR * earners_last),
                                 GROWTH_CAP * earners_last) - 0.5
            rate_struct = 100.0 * max(earners_target, 0.0) / tracked_target
            y_struct = logit(rate_struct)
        else:
            y_struct = y_trend

        median_lg = W_STRUCTURAL * y_struct + (1.0 - W_STRUCTURAL) * y_trend
        delta = median_lg - last_lg
        # An upward move has less room the higher the rate already sits:
        # a trophy at 90% cannot climb the way one at 20% can. A fall has
        # no such ceiling, so the downward cap stays whole.
        up_cap = max_move * (1.0 - 0.5 * (last_rate / 100.0))
        median_lg = last_lg + max(-max_move, min(up_cap, delta * damp))

        sigma = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2 + se_trend ** 2)
        if not refresh_drops:
            # No refresh seen: bimodal risk, price it as a mixture.
            sigma_mix = max(sigma, 0.15)
            m0 = median_lg
            m1 = median_lg - refresh_drop
            qs = []
            for tau in LEVELS:
                lo, hi = m0 - refresh_drop - 8 * sigma_mix, m0 + 8 * sigma_mix
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    f = (1.0 - p_refresh) * norm_cdf((mid - m0) / sigma_mix) + p_refresh * norm_cdf((mid - m1) / sigma_mix)
                    if f < tau:
                        lo = mid
                    else:
                        hi = mid
                qs.append(inv_logit(0.5 * (lo + hi)))
        else:
            sigma = math.sqrt(sigma ** 2 + sigma_refresh ** 2)
            sigma = min(max(sigma, 0.15), 2.5)
            qs = [inv_logit(median_lg + z * sigma) for z in Z]
        rows.append({"id": trophy["id"], "q": [round4(min(99.99, max(0.01, q))) for q in qs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
