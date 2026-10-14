"""Aggregate scores by condition. Works for numeric and labelled traits alike.

Nothing here knows which trait produced the scores.
"""


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
