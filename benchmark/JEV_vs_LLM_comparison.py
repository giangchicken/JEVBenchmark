"""logic · how each model did on one set, and JEV against each LLM on the same samples.

**One model's figures**, from its predictions alone. A prediction with no label -- a parse error
or a failed call -- is wrong, and is counted under `NO_ANSWER` in the confusion matrix, so every
model is scored on the same n. The two failures are also reported apart: `parse_error_rate` is
an answer that could not be read, `error_rate` a call that did not come back. The classes of
macro-F1 are every gold label and every label predicted; `NO_ANSWER` is not one. JEV's
calibration is read off the predictions that carry `probabilities` and `confidence`; a model
without them gets `None` for both.

**JEV against one LLM**, paired on `id`: n₁₀ is the samples only JEV got right, n₀₁ those
only the LLM did, and nothing else moves the comparison. With n = the set's size, π̂_d = (n₁₀ + n₀₁)/n and Δ̂ = (n₁₀ − n₀₁)/n:

- p-value, McNemar exact: `min(1, 2·BinomCDF(min(n₁₀, n₀₁); n₁₀ + n₀₁, ½))`;
- 95% interval of Δ: Δ̂ ± z₀.₉₇₅·√((π̂_d − Δ̂²)/n);
- power, the chance of concluding correctly at this n when the true gap is Δ:
  `Φ((|Δ|·√n − z₁₋α/₂·√π_d) / √(π_d − Δ²))`, with π_d estimated by π̂_d.

The degenerate cases are answered, not divided by zero: no discordant pair gives p = 1 and no
power (π̂_d = 0 says nothing about how often the two would disagree); a gap wider than π̂_d cannot
happen with that many disagreements, so it has no power either; and π_d = Δ², where every
disagreement goes one way, has no variance left, so the test either always rejects or never does.
"""

import math
from collections import Counter
from collections.abc import Sequence
from statistics import NormalDist
from typing import Final

from pydantic import Field

from .schema import Frozen, Prediction

NO_ANSWER: Final = "<no answer>"
ECE_BINS: Final = 10
CONFIDENCE_THRESHOLDS: Final = (0.0, 0.2, 0.4, 0.6, 0.8)
# The power at which a test that found nothing says a gap that large is unlikely, not untested.
ADEQUATE_POWER: Final = 0.8
MEDIAN, P95 = 0.5, 0.95


class ThresholdAccuracy(Frozen):
    """Accuracy over the predictions JEV was at least this confident about."""

    threshold: float
    coverage: float  # the share of the set kept
    accuracy: float | None  # `None` when nothing is kept


class ModelMetrics(Frozen):
    n: int = Field(gt=0)
    accuracy: float
    macro_f1: float
    f1_by_label: dict[str, float]
    confusion: dict[str, dict[str, int]]  # gold -> predicted (or `NO_ANSWER`) -> count
    parse_error_rate: float
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    ece: float | None = None
    accuracy_by_confidence: tuple[ThresholdAccuracy, ...] | None = None


def compute_f1_by_label(predictions: Sequence[Prediction]) -> dict[str, float]:
    """F1 of each class, gold or predicted; 0 for a class with no true positive."""
    labels = sorted(
        {prediction.label for prediction in predictions}
        | {
            prediction.prediction
            for prediction in predictions
            if prediction.prediction is not None
        }
    )
    scores = {}
    for label in labels:
        true_positive = sum(
            prediction.label == label and prediction.prediction == label
            for prediction in predictions
        )
        predicted = sum(prediction.prediction == label for prediction in predictions)
        actual = sum(prediction.label == label for prediction in predictions)
        scores[label] = (
            2 * true_positive / (predicted + actual) if true_positive else 0.0
        )
    return scores


def count_confusion(predictions: Sequence[Prediction]) -> dict[str, dict[str, int]]:
    """Gold label -> what was predicted for it -> how often, both sorted."""
    counts = Counter(
        (prediction.label, prediction.prediction or NO_ANSWER)
        for prediction in predictions
    )
    return {
        gold: {
            predicted: count
            for (row, predicted), count in sorted(counts.items())
            if row == gold
        }
        for gold in sorted({gold for gold, _ in counts})
    }


def compute_percentile(values: Sequence[float], share: float) -> float:
    """The nearest-rank percentile: the smallest value at or above that share of them."""
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)), 1) - 1]


def compute_ece(predictions: Sequence[Prediction]) -> float | None:
    """Expected calibration error over `ECE_BINS` equal bins of the top probability.

    Only predictions that carry a distribution are binned, and weighted by how many they are;
    `None` when none does.
    """
    scored = [
        (max(prediction.probabilities.values()), prediction.correct)
        for prediction in predictions
        if prediction.probabilities
    ]
    if not scored:
        return None
    bins: dict[int, list[tuple[float, bool]]] = {}
    for top, correct in scored:
        bins.setdefault(min(int(top * ECE_BINS), ECE_BINS - 1), []).append(
            (top, correct)
        )
    return sum(
        abs(
            sum(top for top, _ in members) / len(members)
            - sum(correct for _, correct in members) / len(members)
        )
        * len(members)
        / len(scored)
        for members in bins.values()
    )


def compute_accuracy_by_confidence(
    predictions: Sequence[Prediction],
) -> tuple[ThresholdAccuracy, ...] | None:
    """For each threshold, the share of the set JEV was that confident about and its accuracy.

    Coverage is over the whole set, so a failed call is a prediction nobody was confident about.
    """
    if all(prediction.confidence is None for prediction in predictions):
        return None
    rows = []
    for threshold in CONFIDENCE_THRESHOLDS:
        kept = [
            prediction
            for prediction in predictions
            if prediction.confidence is not None and prediction.confidence >= threshold
        ]
        rows.append(
            ThresholdAccuracy(
                threshold=threshold,
                coverage=len(kept) / len(predictions),
                accuracy=sum(prediction.correct for prediction in kept) / len(kept)
                if kept
                else None,
            )
        )
    return tuple(rows)


def compute_metrics(predictions: Sequence[Prediction]) -> ModelMetrics:
    """Every R11 figure for one model on one set. `ValueError` for no predictions."""
    if not predictions:
        raise ValueError("no predictions to score")
    n = len(predictions)
    f1_by_label = compute_f1_by_label(predictions)
    latencies = [prediction.latency_ms for prediction in predictions]
    return ModelMetrics(
        n=n,
        accuracy=sum(prediction.correct for prediction in predictions) / n,
        macro_f1=sum(f1_by_label.values()) / len(f1_by_label),
        f1_by_label=f1_by_label,
        confusion=count_confusion(predictions),
        parse_error_rate=sum(
            prediction.prediction is None and prediction.error is None
            for prediction in predictions
        )
        / n,
        error_rate=sum(prediction.error is not None for prediction in predictions) / n,
        latency_p50_ms=compute_percentile(latencies, MEDIAN),
        latency_p95_ms=compute_percentile(latencies, P95),
        ece=compute_ece(predictions),
        accuracy_by_confidence=compute_accuracy_by_confidence(predictions),
    )


# ---------- JEV against one LLM ----------
DEFAULT_ALPHA: Final = 0.05
CONFIDENCE_LEVEL: Final = 0.95
GAPS: Final = (0.05, 0.10, 0.15, 0.20)
STANDARD_NORMAL: Final = NormalDist()


class PairedComparison(Frozen):
    jev: str
    llm: str
    n: int
    n10: int  # only JEV right
    n01: int  # only the LLM right
    delta: float  # JEV's accuracy minus the LLM's
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float
    power_at_observed: float | None
    power_at_gap: dict[
        str, float | None
    ]  # "0.10" -> power to detect a gap of 10 points
    conclusion: str


def compute_mcnemar_p(n10: int, n01: int) -> float:
    """The exact two-sided McNemar p-value; 1 when the two never disagree."""
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, wins) for wins in range(min(n10, n01) + 1))
    return min(1.0, 2 * tail / (1 << discordant))


def compute_power(gap: float, discordant: float, n: int, alpha: float) -> float | None:
    """The chance a two-sided McNemar at `alpha` detects a true gap, given π_d = `discordant`."""
    gap = abs(gap)
    if discordant <= 0 or gap > discordant:
        return None
    critical = STANDARD_NORMAL.inv_cdf(1 - alpha / 2) * math.sqrt(discordant)
    variance = discordant - gap**2
    if variance <= 0:
        pairs = round(discordant * n)
        return 1.0 if compute_mcnemar_p(pairs, 0) < alpha else 0.0
    return STANDARD_NORMAL.cdf((gap * math.sqrt(n) - critical) / math.sqrt(variance))


def write_conclusion(
    llm: str,
    n: int,
    delta: float,
    discordant: float,
    p_value: float,
    alpha: float,
    power_at_ten: float | None,
) -> str:
    """One Vietnamese sentence a reader can act on."""
    points = abs(delta) * 100
    if p_value < alpha:
        better = "cao hơn" if delta > 0 else "thấp hơn"
        return (
            f"JEV có accuracy {better} {llm} {points:.1f} điểm, khác biệt có ý nghĩa thống kê "
            f"(p = {p_value:.3g} < {alpha})."
        )
    if power_at_ten is not None and power_at_ten >= ADEQUATE_POWER:
        power = (
            f"xác suất phát hiện một chênh lệch 10 điểm là {power_at_ten:.0%}, nên nếu hai bên"
            " có khác nhau thì nhiều khả năng dưới 10 điểm"
        )
    else:
        power = (
            f"xác suất phát hiện một chênh lệch 10 điểm chỉ là {power_at_ten:.0%}"
            if power_at_ten is not None
            else (
                "không ước lượng được xác suất phát hiện một chênh lệch 10 điểm, vì hai model chỉ"
                f" bất đồng ở {discordant:.0%} số mẫu"
            )
        )
    return (
        f"Chưa kết luận được JEV khác {llm} (Δ = {delta * 100:+.1f} điểm, p = {p_value:.3g}); "
        f"với n = {n}, {power}."
    )


def compare_paired(
    jev: str,
    jev_predictions: Sequence[Prediction],
    llm: str,
    llm_predictions: Sequence[Prediction],
    alpha: float = DEFAULT_ALPHA,
) -> PairedComparison:
    """The comparison, or `ValueError` when the two did not answer the same samples."""
    jev_correct = {prediction.id: prediction.correct for prediction in jev_predictions}
    llm_correct = {prediction.id: prediction.correct for prediction in llm_predictions}
    if (
        jev_correct.keys() != llm_correct.keys()
        or len(jev_correct) != len(jev_predictions)
        or len(llm_correct) != len(llm_predictions)
    ):
        raise ValueError(f"{jev} and {llm} were not run on the same set of ids")
    n = len(jev_correct)
    if n == 0:
        raise ValueError("no samples to compare")
    n10 = sum(
        jev_correct[sample_id] and not llm_correct[sample_id]
        for sample_id in jev_correct
    )
    n01 = sum(
        llm_correct[sample_id] and not jev_correct[sample_id]
        for sample_id in jev_correct
    )
    discordant = (n10 + n01) / n
    delta = (n10 - n01) / n
    p_value = compute_mcnemar_p(n10, n01)
    half_width = STANDARD_NORMAL.inv_cdf(1 - (1 - CONFIDENCE_LEVEL) / 2) * math.sqrt(
        max(discordant - delta**2, 0.0) / n
    )
    power_at_gap = {
        f"{gap:.2f}": compute_power(gap, discordant, n, alpha) for gap in GAPS
    }
    return PairedComparison(
        jev=jev,
        llm=llm,
        n=n,
        n10=n10,
        n01=n01,
        delta=delta,
        p_value=p_value,
        ci_low=delta - half_width,
        ci_high=delta + half_width,
        alpha=alpha,
        power_at_observed=compute_power(delta, discordant, n, alpha)
        if n10 + n01
        else None,
        power_at_gap=power_at_gap,
        conclusion=write_conclusion(
            llm, n, delta, discordant, p_value, alpha, power_at_gap[f"{0.10:.2f}"]
        ),
    )
