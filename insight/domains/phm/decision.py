# -*- coding: utf-8 -*-
"""Auditable PHM candidate calculations using only NumPy and the standard library.

These calculations do not authorize maintenance or shutdown.  In particular,
Gaussian risk probabilities remain an approximation until sigma is calibrated.
"""
import math

import numpy as np


def _number(value, name, positive=False):
    """Validate configuration before calculating even an empty fleet."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be a finite number" % name)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s must be a finite number" % name)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        bound = "positive" if positive else "nonnegative"
        raise ValueError("%s must be finite and %s" % (name, bound))
    return number


def _vector(df, column):
    try:
        values = np.asarray(df[column], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing or nonnumeric column: %s" % column)
    if values.ndim != 1 or len(values) != len(df) or not np.isfinite(values).all():
        raise ValueError("%s must contain one finite value per unit" % column)
    return values


def _unit_ids(df, required=False):
    if "unit" not in df:
        if required and len(df):
            raise ValueError("missing column: unit")
        return np.arange(1, len(df) + 1, dtype=int)
    values = _vector(df, "unit")
    if (values < 1).any() or (values != np.floor(values)).any():
        raise ValueError("unit must contain positive integer identifiers")
    if len(np.unique(values)) != len(values):
        raise ValueError("unit identifiers must be unique")
    return values


def _risk_probabilities(pred, sigma, horizon):
    """P(RUL < H), with a strict event for a zero-variance distribution."""
    probabilities = []
    for mu, sd in zip(pred, sigma):
        if sd == 0:
            probabilities.append(float(mu < horizon))
        else:
            # erfc retains Gaussian tails where 1 + erf would round to zero.
            # Subtract first to preserve close means.  If that overflows, H is
            # nonnegative and mu is negative: dividing the two terms separately
            # cannot produce inf - inf and preserves a finite standardized gap.
            delta = float(mu) - horizon
            if math.isfinite(delta):
                z = (delta / float(sd)) / math.sqrt(2.0)
            else:
                z = ((float(mu) / float(sd)) / math.sqrt(2.0)
                     - (horizon / float(sd)) / math.sqrt(2.0))
            probabilities.append(0.5 * math.erfc(z))
    return np.asarray(probabilities, dtype=float)


def _total_cost(pred, risks, threshold, c_fn, c_fp):
    alarms = pred <= threshold
    with np.errstate(over="ignore", invalid="ignore"):
        cost = float(c_fn * np.sum(risks[~alarms]) + c_fp * np.count_nonzero(alarms))
    if not math.isfinite(cost):
        raise ValueError("total expected cost is not finite for these inputs")
    return cost, int(np.count_nonzero(alarms))


def _optimum(pred, risks, grid, c_fn, c_fp):
    curve = []
    for threshold in grid:
        cost, n_alarms = _total_cost(pred, risks, threshold, c_fn, c_fp)
        curve.append({"threshold_rul": threshold, "expected_cost": cost,
                      "n_alarms": n_alarms})
    # Exact ties select the lower threshold.  No tolerance can hide a real cost
    # difference or make the result depend on grid iteration order.
    best = min(curve, key=lambda row: (row["expected_cost"], row["threshold_rul"]))
    return best, curve


def cost_threshold(df, c_fn=10, c_fp=1, risk_horizon_cycles=30,
                   default_threshold=30, thresholds=None):
    """Choose a mean-RUL alarm line against a *fixed* failure-risk horizon H.

    For a fleet, E(t) = C_fn * sum(P(RUL_i < H) for nonalarms)
    + C_fp * count(alarms), where alarm_i(t) = (mu_i <= t).
    Changing H with t makes an all-clear threshold degenerate, so only the
    alarm line is searched.  Costs are fleet totals in assumed cost units.
    User thresholds augment a grid covering every distinct alarm set; the
    default threshold is always included.  Negative predictions are preserved
    for the RUL gate and already alarm at the lowest permitted threshold, 0.
    """
    c_fn = _number(c_fn, "c_fn")
    c_fp = _number(c_fp, "c_fp", positive=True)
    horizon = _number(risk_horizon_cycles, "risk_horizon_cycles")
    default = _number(default_threshold, "default_threshold")
    ratio = c_fn / c_fp
    if not math.isfinite(ratio):
        raise ValueError("c_fn / c_fp must be finite")
    pred = _vector(df, "pred") if len(df) else np.asarray([], dtype=float)
    sigma = _vector(df, "sigma") if len(df) else np.asarray([], dtype=float)
    if (sigma < 0).any():
        raise ValueError("sigma must be nonnegative")
    units = _unit_ids(df)
    risks = _risk_probabilities(pred, sigma, horizon)

    grid = {0.0, 30.0, default}
    grid.update(float(mu) for mu in pred if mu >= 0)
    if len(pred):
        maximum = max(0.0, float(np.max(pred)))
        # At float's largest finite value, maxmu itself already alarms all units.
        upper = maximum + 1.0
        if math.isfinite(upper):
            grid.add(upper)
    if thresholds is not None:
        if isinstance(thresholds, (str, bytes)):
            raise ValueError("thresholds must be an iterable of numeric values")
        try:
            grid.update(_number(value, "thresholds") for value in thresholds)
        except TypeError:
            raise ValueError("thresholds must be an iterable of finite nonnegative values")
    grid = sorted(grid)
    best, curve = _optimum(pred, risks, grid, c_fn, c_fp)
    default_cost, _ = _total_cost(pred, risks, default, c_fn, c_fp)

    ratios = sorted({0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, ratio})
    ratio_curve = []
    for value in ratios:
        # Unit false-positive cost avoids overflow of value*c_fp and leaves the
        # optimum unchanged, since all costs have the same positive scale.
        optimum, _ = _optimum(pred, risks, grid, value, 1.0)
        ratio_curve.append({"cost_ratio": value, "threshold_rul": optimum["threshold_rul"]})
    violations = []
    for previous, current in zip(ratio_curve, ratio_curve[1:]):
        if current["threshold_rul"] < previous["threshold_rul"]:
            violations.append({"lower_cost_ratio": previous["cost_ratio"],
                               "higher_cost_ratio": current["cost_ratio"],
                               "previous_threshold": previous["threshold_rul"],
                               "threshold_rul": current["threshold_rul"]})
    return {
        "n": int(len(pred)), "threshold_rul": best["threshold_rul"],
        "expected_cost": best["expected_cost"], "default_cost": default_cost,
        "compare_to_default": default_cost - best["expected_cost"],
        "default_threshold": default, "n_alarms": best["n_alarms"],
        "risk_horizon_cycles": horizon, "c_fn": c_fn, "c_fp": c_fp,
        "cost_ratio": ratio, "threshold_curve": curve, "ratio_curve": ratio_curve,
        "threshold_violations": violations,
        "bad_units": [int(unit) for unit in units[pred < 0]],
        "cost_unit": "assumed cost units, fleet total",
        "assumptions": {
            "risk_distribution": "RUL ~ N(pred, sigma); uncalibrated approximation",
            "risk_event": "RUL < risk_horizon_cycles (fixed during threshold search)",
            "alarm_rule": "pred <= threshold_rul", "c_fn": c_fn, "c_fp": c_fp,
            "risk_horizon_cycles": horizon,
        },
    }


def maintenance_plan(df, lead_time_days=14, window_days=14, cycles_per_day=1):
    """Return a point-estimate candidate order list, never shutdown authority.

    service_day = pred / cycles_per_day; only service_day <= window_days is
    included.  Missed lead times are determined *before* order day is clipped
    to today.  Negative predictions remain visible and missed, for gate review.
    """
    lead = _number(lead_time_days, "lead_time_days")
    window = _number(window_days, "window_days")
    rate = _number(cycles_per_day, "cycles_per_day", positive=True)
    pred = _vector(df, "pred") if len(df) else np.asarray([], dtype=float)
    units = _unit_ids(df, required=True)
    with np.errstate(over="ignore", invalid="ignore"):
        days = pred / rate
        raw_days = days - lead
    if not np.isfinite(days).all() or not np.isfinite(raw_days).all():
        raise ValueError("converted service and order days must be finite")
    indices = sorted(range(len(pred)), key=lambda i: (float(pred[i]), int(units[i])))
    orders = []
    for i in indices:
        if days[i] > window:
            continue
        raw = float(raw_days[i])
        orders.append({"unit": int(units[i]), "pred_rul": float(pred[i]),
                       "service_day": float(days[i]), "raw_order_day": raw,
                       "order_day": max(0.0, raw), "missed": bool(raw < 0)})
    missed = [row["unit"] for row in orders if row["missed"]]
    return {
        "n": int(len(pred)), "n_orders": len(orders), "n_missed": len(missed),
        "earliest_unit": orders[0]["unit"] if orders else 0,
        "latest_order_day": max((row["order_day"] for row in orders), default=0.0),
        "orders": orders, "missed_units": missed,
        "bad_units": [int(unit) for unit in units[pred < 0]],
        "lead_time_days": lead, "window_days": window, "cycles_per_day": rate,
        "assumptions": {"lead_time_days": lead, "window_days": window,
                        "cycles_per_day": rate,
                        "schedule_basis": "point-estimate candidate list for human review"},
    }
