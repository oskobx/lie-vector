"""Aggregate scores by condition. Works for numeric and labelled traits alike.

Nothing here knows which trait produced the scores.
"""

import math

import numpy as np

from trait_vectors import config


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion k/n: better than p ± 1.96·se when p is near 0 or 1.

    It is the set of true rates p for which the observed k/n lies within z
    standard errors of p, solved in closed form.
    """
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)  # clamp float rounding (e.g. -0.0%)


def auroc(pos, neg):
    """Area under the ROC curve: P(score of a random pos item > score of a random neg item), ties count 1/2.

    0.5 = no separation, 1.0 = every pos score above every neg score, 0.0 = all below.
    Computed over all len(pos)·len(neg) pairs, which is fine for hundreds of items.
    """
    p = np.asarray(pos, dtype=float)[:, None]  # column: one row per pos score
    n = np.asarray(neg, dtype=float)[None, :]  # row: one column per neg score
    return float((p > n).mean() + 0.5 * (p == n).mean())


def stratified_auroc(scores, positive, strata, min_per_side=5):
    """Weighted mean of the AUROC within each stratum (Phase 3 spec §4).

    scores[i] is item i's score, positive[i] its label (True = positive, e.g. a
    lie), strata[i] its stratum (e.g. a level). Within stratum s, with n_P
    positive and n_N negative items, the AUROC compares only items of s; the
    strata are averaged with weights w_s = n_P n_N / (n_P + n_N), half the
    harmonic mean of the two counts, as for v_honesty. A stratum with fewer
    than min_per_side items on either side is skipped. Adding a constant to
    every score of one stratum changes nothing.
    """
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    strata = np.asarray(strata)
    total, weight_sum = 0.0, 0.0
    for s in np.unique(strata):
        here = strata == s  # boolean mask: True for the items of stratum s
        pos, neg = scores[here & positive], scores[here & ~positive]
        if min(len(pos), len(neg)) < min_per_side:
            continue
        w = len(pos) * len(neg) / (len(pos) + len(neg))
        total += w * auroc(pos, neg)
        weight_sum += w
    if weight_sum == 0:
        raise ValueError(f"no stratum has {min_per_side} items on both sides")
    return total / weight_sum


def paired_bootstrap(a, b, n_draws=2000, seed=config.SEED):
    """(mean, lo, hi): mean of a - b and its paired bootstrap 95% interval (Phase 2 spec §1).

    a[i] and b[i] are two measurements of the same scenario i (e.g. lie = 1 / 0
    at two α values). The differences d_i = a[i] - b[i] are resampled with
    replacement, n at a time, n_draws times; the interval is the 2.5th and
    97.5th percentile of the resampled means. Pairing by scenario cancels how
    tempting each scenario is in itself, which narrows the interval.
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} vs {len(b)}")
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    # Row j of idx holds the n scenario indices of draw j; d[idx] picks those
    # differences (numpy indexing with an integer array), and .mean(axis=1)
    # averages each row, giving one resampled mean per draw.
    idx = rng.integers(0, len(d), size=(n_draws, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def summarize(rows):
    """Summarise (condition, Score) pairs, one summary per condition.

    Returns {condition: summary}, where summary has
        "n"      number of scores for that condition
        "mean"   mean of Score.value, if any score has a value
        "rates"  {label: fraction of labelled scores}, if any score has a label
    Confidence intervals come later, with the code that reports them.
    """
    by_condition = {}
    for condition, score in rows:
        by_condition.setdefault(condition, []).append(score)

    out = {}
    for condition, scores in by_condition.items():
        summary = {"n": len(scores)}
        values = [s.value for s in scores if s.value is not None]
        labels = [s.label for s in scores if s.label is not None]
        if values:
            summary["mean"] = sum(values) / len(values)
        if labels:
            summary["rates"] = {lab: labels.count(lab) / len(labels) for lab in sorted(set(labels))}
        out[condition] = summary
    return out
