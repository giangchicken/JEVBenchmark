"""wiring · one dataset through every model, into `results/<dataset>/<run_id>/`.

    uv run python -m benchmark.main --dataset intent [--models gemma4-31B-it jev-latest]
        [--run-id 20260925-101500] [--alpha 0.05] [--limit 5]

Every config is read and checked before the first call, so a broken one stops the run before
anything is spent. Each model gets its own semaphore of `max_concurrency` and the models run side
by side. A sample is appended to `predictions/<model>.jsonl` as soon as it comes back.

**Resume** (R14): a run started again with the same `--run-id` asks nothing a model already
answered. An answer that could not be parsed counts as answered -- asking again would give that
model a second try the others did not get -- and only a failed call is asked again.

`--dataset all` is each set in turn under one run id, each with its own report; nothing is
merged. Without a JEV config the LLMs still run and the report says there is nothing to pair.
"""

import argparse
import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from agent_toolkit.logging import get_logger

from . import schema, tasks
from .events import install_structured_events
from .JEV_vs_LLM_comparison import DEFAULT_ALPHA, compare_paired, compute_metrics
from .models import CONFIG_DIR, ConfigError, ModelConfig, load_model_configs
from .report_generation import (
    ModelResult,
    Report,
    append_prediction,
    build_predictions_path,
    read_prediction_file,
    write_report,
)
from .runners import build_jev_model, predict_with_jev, predict_with_llm
from .schema import Prediction, Task

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "test"
RESULTS_DIR = ROOT / "results"
ALL: Final = "all"

DATASETS: Final[dict[str, tuple[type[Any], Task]]] = {
    "data_collection": (schema.DataCollectionTestSet, tasks.data_collection_prediction),
    "intent": (schema.IntentTestSet, tasks.intent_prediction),
    "tool_decision": (schema.ToolDecisionTestSet, tasks.tool_decision_prediction),
    "gender": (schema.GenderTestSet, tasks.gender_prediction),
    "sentiment": (schema.SentimentTestSet, tasks.sentiment_prediction),
}

# One sample through one model. The label goes along only to be copied onto the prediction.
Predictor = Callable[[ModelConfig, Task, Any], Awaitable[Prediction]]

# Named, not `__name__`: run as `-m benchmark.main` that is `__main__`, outside the tree
# `install_structured_events` listens on.
logger = get_logger("benchmark.main")


def build_predictor(configs: Sequence[ModelConfig]) -> Predictor:
    """The real one: JEV for a typesafe config, `complete` for any other."""
    jev_models = {
        config.name: build_jev_model(config) for config in configs if config.is_jev
    }

    async def predict(config: ModelConfig, task: Task, sample: Any) -> Prediction:
        if config.is_jev:
            return await predict_with_jev(
                config,
                jev_models[config.name],
                sample.id,
                sample.label,
                task.build_jev_input(sample.id, sample.input),
            )
        return await predict_with_llm(
            config,
            sample.id,
            sample.label,
            task.build_llm_messages(sample.id, sample.input),
            lambda answer: task.parse_llm_answer(sample.id, sample.input, answer),
        )

    return predict


def load_samples(
    dataset: str, limit: int | None, data_dir: Path = DATA_DIR
) -> tuple[Any, ...]:
    """The set's samples in file order, the first `limit` of them when one is given."""
    test_set = DATASETS[dataset][0]
    path = data_dir / f"{dataset}.json"
    samples = tuple(test_set.model_validate_json(path.read_text(encoding="utf-8")).root)
    return samples[:limit] if limit is not None else samples


async def run_model(
    config: ModelConfig,
    task: Task,
    samples: Sequence[Any],
    path: Path,
    predict: Predictor,
) -> list[Prediction]:
    """Every sample's prediction by that model, asking only what the file lacks or failed.

    A stored prediction also counts as missing when its gold label is not the sample's: the
    test set was corrected since, and scoring it against the old label would be silently wrong.
    """
    labels = {sample.id: sample.label for sample in samples}
    done = {
        sample_id
        for sample_id, prediction in read_prediction_file(path).items()
        if prediction.error is None and prediction.label == labels.get(sample_id)
    }
    pending = [sample for sample in samples if sample.id not in done]
    gate = asyncio.Semaphore(config.max_concurrency)

    async def ask(sample: Any) -> None:
        async with gate:
            prediction = await predict(config, task, sample)
        append_prediction(path, prediction)

    await asyncio.gather(*(ask(sample) for sample in pending))
    logger.info(
        "model_finished",
        extra={
            "model": config.name,
            "asked": len(pending),
            "resumed": len(samples) - len(pending),
        },
    )
    found = read_prediction_file(path)
    unreadable = [sample.id for sample in samples if sample.id not in found]
    if unreadable:
        raise RuntimeError(f"{path}: no readable prediction for {unreadable}")
    return [found[sample.id] for sample in samples]


async def run_dataset(
    dataset: str,
    configs: Sequence[ModelConfig],
    run_dir: Path,
    predict: Predictor,
    alpha: float = DEFAULT_ALPHA,
    limit: int | None = None,
    data_dir: Path = DATA_DIR,
) -> Report:
    """Run that set through every model, score it, pair JEV with each LLM, write the report."""
    task = DATASETS[dataset][1]
    samples = load_samples(dataset, limit, data_dir)
    logger.info(
        "dataset_started",
        extra={
            "dataset": dataset,
            "run_dir": str(run_dir),
            "n": len(samples),
            "models": [config.name for config in configs],
        },
    )
    runs = await asyncio.gather(
        *(
            run_model(
                config,
                task,
                samples,
                build_predictions_path(run_dir, config.name),
                predict,
            )
            for config in configs
        )
    )
    predictions = {config.name: run for config, run in zip(configs, runs, strict=True)}

    warnings: list[str] = []
    jevs = [config for config in configs if config.is_jev]
    llms = [config for config in configs if not config.is_jev]
    if not jevs:
        warnings.append(
            "không có config JEV (`provider: typesafe`), nên không có phần ghép cặp"
        )
        logger.warning("jev_missing", extra={"dataset": dataset})

    models = []
    for config in configs:
        metrics = compute_metrics(predictions[config.name])
        served = sorted(
            {
                prediction.served_model
                for prediction in predictions[config.name]
                if prediction.served_model
            }
        )
        result = ModelResult(
            name=config.name,
            is_jev=config.is_jev,
            served_models=tuple(served),
            metrics=metrics,
        )
        if result.is_unreliable:
            warnings.append(
                f"{config.name}: {metrics.error_rate:.0%} số lần gọi bị lỗi"
            )
            logger.warning(
                "high_error_rate",
                extra={"model": config.name, "error_rate": metrics.error_rate},
            )
        models.append(result)

    comparisons = tuple(
        compare_paired(
            jev.name, predictions[jev.name], llm.name, predictions[llm.name], alpha
        )
        for jev in jevs
        for llm in llms
    )
    report = Report(
        dataset=dataset,
        run_id=run_dir.name,
        n=len(samples),
        models=tuple(models),
        comparisons=comparisons,
        warnings=tuple(warnings),
    )
    path = write_report(run_dir, report)
    logger.info("report_written", extra={"dataset": dataset, "path": str(path)})
    return report


def read_positive_int(text: str) -> int:
    """An argparse type: a whole number above 0, or the error argparse shows the user."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0, got {value}")
    return value


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmark.main")
    parser.add_argument("--dataset", required=True, choices=[*DATASETS, ALL])
    parser.add_argument(
        "--models", nargs="+", help="config file stems; default: every one"
    )
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--limit",
        type=read_positive_int,
        help="only the first N samples, for a trial run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    try:
        configs = load_model_configs(CONFIG_DIR, arguments.models)
    except ConfigError as error:
        raise SystemExit(f"config error: {error}") from error
    install_structured_events(logging.INFO)
    datasets = list(DATASETS) if arguments.dataset == ALL else [arguments.dataset]
    asyncio.run(run_datasets(datasets, configs, arguments))


async def run_datasets(
    datasets: Sequence[str],
    configs: Sequence[ModelConfig],
    arguments: argparse.Namespace,
) -> None:
    """Each set in turn, in one event loop: a JEV client is bound to the loop it first ran in."""
    predict = build_predictor(configs)
    for dataset in datasets:
        await run_dataset(
            dataset,
            configs,
            RESULTS_DIR / dataset / arguments.run_id,
            predict,
            arguments.alpha,
            arguments.limit,
        )


if __name__ == "__main__":
    main()
