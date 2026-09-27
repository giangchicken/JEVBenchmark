"""adapter · a run's files: `predictions/<model>.jsonl`, `report.json` and `report.md`.

A run directory is `results/<dataset>/<run_id>/`. Predictions are appended one line per sample
as each one comes back, so a run that stops keeps what it had; reading them back keeps the
last line per id and skips a line that is not a whole prediction -- the one a crash cut short.

The report is written twice from one `Report`: as JSON for a program, as Markdown for a reader.
No key is in either: a `Report` holds names, figures and sentences, never a config.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from agent_toolkit.file_utils import write_json
from pydantic import ValidationError

from .JEV_vs_LLM_comparison import ModelMetrics, PairedComparison
from .schema import Frozen, Prediction

PREDICTIONS_DIR: Final = "predictions"
HIGH_ERROR_RATE: Final = (
    0.20  # above this a model's figures are mostly about its endpoint
)
WARNING_MARK: Final = "⚠"


class ModelResult(Frozen):
    name: str
    is_jev: bool
    served_models: tuple[str, ...]  # what the endpoint said answered, e.g. `jev-1.13.0`
    metrics: ModelMetrics

    @property
    def is_unreliable(self) -> bool:
        return self.metrics.error_rate > HIGH_ERROR_RATE


class Report(Frozen):
    dataset: str
    run_id: str
    n: int
    models: tuple[ModelResult, ...]
    comparisons: tuple[PairedComparison, ...]
    warnings: tuple[str, ...]


def build_predictions_path(run_dir: Path, model: str) -> Path:
    return run_dir / PREDICTIONS_DIR / f"{model}.jsonl"


def append_prediction(path: Path, prediction: Prediction) -> None:
    """One more line, flushed, so it survives the run stopping right after.

    A file a crash left without its final newline gets one first, or the new line would be
    glued to the cut one and lost with it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab+") as file:  # append mode: every write lands at the end
        if file.seek(0, os.SEEK_END):
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b"\n":
                file.write(b"\n")
        file.write(prediction.model_dump_json().encode() + b"\n")
        file.flush()


def read_prediction_file(path: Path) -> dict[str, Prediction]:
    """The last prediction per id in that file; `{}` when there is no file yet.

    Not `read_jsonlines`, which gives up on the whole file at one bad line: a line cut short by a
    crash would make a resumed run ask every sample again.
    """
    if not path.exists():
        return {}
    found: dict[str, Prediction] = {}
    # `split("\n")`, not `splitlines()`: that also breaks on U+2028 and `\x85`, which a JSON
    # string may hold raw, and would cut one prediction into two unreadable halves.
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            prediction = Prediction.model_validate_json(line)
        except ValidationError:
            continue
        found[prediction.id] = prediction
    return found


def format_share(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def format_row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def format_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [format_row(header), format_row(["---"] * len(header))] + [
        format_row(row) for row in rows
    ]


def render_summary(report: Report) -> list[str]:
    rows = [
        [
            f"{WARNING_MARK} {model.name}" if model.is_unreliable else model.name,
            "JEV" if model.is_jev else "LLM",
            ", ".join(model.served_models) or "—",
            model.metrics.n,
            format_share(model.metrics.accuracy),
            format_share(model.metrics.macro_f1),
            format_share(model.metrics.parse_error_rate),
            format_share(model.metrics.error_rate),
            f"{model.metrics.latency_p50_ms:.0f}",
            f"{model.metrics.latency_p95_ms:.0f}",
        ]
        for model in report.models
    ]
    header = ["model", "loại", "phiên bản", "n", "accuracy", "macro-F1"]
    header += ["parse lỗi", "gọi lỗi", "p50 ms", "p95 ms"]
    lines = ["## Tổng hợp theo model", "", *format_table(header, rows)]
    if any(model.is_unreliable for model in report.models):
        lines += [
            "",
            f"{WARNING_MARK} tỉ lệ gọi lỗi trên {HIGH_ERROR_RATE:.0%}: số liệu của model này"
            " phản ánh endpoint nhiều hơn năng lực của model.",
        ]
    return lines


def render_comparisons(report: Report) -> list[str]:
    if not report.comparisons:
        return []
    gaps = list(report.comparisons[0].power_at_gap)
    header = ["JEV vs", "n₁₀", "n₀₁", "Δ̂", "KTC 95%", "p", "power tại Δ̂"]
    header += [f"power Δ={gap}" for gap in gaps]
    rows = [
        [
            comparison.llm,
            comparison.n10,
            comparison.n01,
            f"{comparison.delta * 100:+.1f}",
            f"[{comparison.ci_low * 100:+.1f}, {comparison.ci_high * 100:+.1f}]",
            f"{comparison.p_value:.3g}",
            format_share(comparison.power_at_observed),
            *(format_share(comparison.power_at_gap[gap]) for gap in gaps),
        ]
        for comparison in report.comparisons
    ]
    return [
        "## Ghép cặp JEV với từng LLM (McNemar exact)",
        "",
        "n₁₀: chỉ JEV đúng. n₀₁: chỉ LLM đúng. Δ, KTC tính bằng điểm phần trăm."
        " Power là xác suất kết luận đúng tại n này; — là chênh lệch lớn hơn tỉ lệ bất"
        " đồng quan sát được, nên không ước lượng được.",
        "",
        *format_table(header, rows),
        "",
        *(
            f"- **{comparison.jev} vs {comparison.llm}**: {comparison.conclusion}"
            for comparison in report.comparisons
        ),
    ]


def render_calibration(report: Report) -> list[str]:
    lines: list[str] = []
    for model in report.models:
        rows = model.metrics.accuracy_by_confidence
        if rows is None:
            continue
        ece = model.metrics.ece
        lines += [
            f"## {model.name}: hiệu chỉnh",
            "",
            f"ECE (10 bin): {'—' if ece is None else f'{ece:.3f}'}",
            "",
            *format_table(
                ["confidence ≥", "phủ", "accuracy"],
                [
                    [
                        row.threshold,
                        format_share(row.coverage),
                        format_share(row.accuracy),
                    ]
                    for row in rows
                ],
            ),
            "",
        ]
    return lines


def render_f1_by_label(report: Report) -> list[str]:
    labels = sorted(
        {label for model in report.models for label in model.metrics.f1_by_label}
    )
    rows = [
        [
            label,
            *(
                format_share(model.metrics.f1_by_label.get(label))
                for model in report.models
            ),
        ]
        for label in labels
    ]
    return [
        "## F1 theo nhãn",
        "",
        *format_table(["nhãn", *(model.name for model in report.models)], rows),
    ]


def render_confusion(model: ModelResult) -> list[str]:
    confusion = model.metrics.confusion
    predicted = sorted({column for row in confusion.values() for column in row})
    rows = [
        [gold, *(row.get(column, 0) for column in predicted)]
        for gold, row in confusion.items()
    ]
    return [
        f"### {model.name}",
        "",
        *format_table(["gold \\ dự đoán", *predicted], rows),
        "",
    ]


def render_report(report: Report) -> str:
    """The report as Markdown: summary, pairs, calibration, F1 by label, confusion matrices."""
    lines = [
        f"# Benchmark `{report.dataset}` · run `{report.run_id}`",
        "",
        f"n = {report.n}",
        "",
    ]
    lines += [f"> {WARNING_MARK} {warning}" for warning in report.warnings]
    lines += ["", *render_summary(report), ""]
    lines += [*render_comparisons(report), ""]
    lines += [*render_calibration(report)]
    lines += [*render_f1_by_label(report), "", "## Confusion matrix", ""]
    for model in report.models:
        lines += render_confusion(model)
    return "\n".join(lines).rstrip() + "\n"


def write_report(run_dir: Path, report: Report) -> Path:
    """`report.json` and `report.md` in that directory; the Markdown one's path."""
    write_json(run_dir / "report.json", json.loads(report.model_dump_json()))
    markdown = run_dir / "report.md"
    markdown.write_text(render_report(report), encoding="utf-8")
    return markdown
