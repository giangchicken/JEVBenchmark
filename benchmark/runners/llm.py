"""adapter · one sample through an OpenAI-compatible LLM, as a `Prediction`.

The call is `agent_toolkit.llm.complete`, retries and all; what is asked and how the answer is
read are the task's, handed in as `messages` and `parse`. A failed call never raises: one that
still fails after the retries is a `Prediction` with `error` set, and one `prediction_failed` event.

`Prediction.input` is `{"complete": {...}}`, the keyword arguments `complete` was called with, the
key left out. Not the request on the wire, as JEV's is: `complete` builds its own OpenAI client
and offers no hook on it, so what it adds on the way -- a default `max_tokens`, `enable_thinking`
moved into `chat_template_kwargs` -- is not in the record.
"""

import time
from collections.abc import Callable
from typing import Any

from agent_toolkit.llm import complete
from agent_toolkit.logging import get_logger

from ..models import ModelConfig
from ..schema import Prediction

logger = get_logger(__name__)


def build_completion_settings(config: ModelConfig) -> dict[str, Any]:
    """The sampling settings that config declares, and only those: an unset one keeps its default."""
    declared = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "enable_thinking": config.enable_reason,
    }
    return {
        **{name: value for name, value in declared.items() if value is not None},
        **config.extra,
    }


async def predict_with_llm(
    config: ModelConfig,
    sample_id: str,
    label: str,
    messages: list[dict[str, object]],
    parse: Callable[[str], str | None],
) -> Prediction:
    """What that model answers to those messages, read by `parse`. `label` is only copied through."""
    arguments: dict[str, Any] = {
        "messages": messages,
        "model": config.model,
        "base_url": config.base_url,
        **build_completion_settings(config),
    }
    sent = {"complete": arguments}
    started = time.perf_counter()
    try:
        answer = await complete(
            "", api_key=config.api_key.get_secret_value(), **arguments
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        logger.warning(
            "prediction_failed",
            extra={"model": config.name, "id": sample_id, "error": message},
        )
        return Prediction(
            id=sample_id,
            label=label,
            input=sent,
            prediction=None,
            error=message,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    return Prediction(
        id=sample_id,
        label=label,
        input=sent,
        prediction=parse(answer),
        latency_ms=(time.perf_counter() - started) * 1000,
        raw_answer=answer,
        served_model=config.model,
    )
