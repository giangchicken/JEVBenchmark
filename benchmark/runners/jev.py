"""adapter · one sample through JEV, as a `Prediction`.

JEV answers a typed question about a text: the task hands in a `JevInput` whose output type has
exactly one field, whose description is the question, and whose state is the text. The pick is
turned into the set's label through `option_to_label`; JEV's distribution and confidence for that
field come from `provider_details`, and the version that answered from the response, because
`jev-latest` is an alias that moves.

A failed call never raises: it is a `Prediction` with `error` set and one `prediction_failed` event;
a pick `option_to_label` does not know is a parse error, as it would be for an LLM.
"""

import time

import httpx2
import pydantic_ai
from agent_toolkit.logging import get_logger
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from typesafe_sdk._core.constants import SYSTEM_ONE_PATH

from ..models import ModelConfig
from ..schema import JevInput, Prediction

logger = get_logger(__name__)

# A first run prints a banner to stdout, which is where the events go.
pydantic_ai.BANNER_ENABLED = False


def build_endpoint_client(
    endpoint: str, transport: httpx2.AsyncBaseTransport | None = None
) -> httpx2.AsyncClient:
    """A client that sends TypeSafe's decisions call to that URL instead.

    The TypeSafe SDK always posts `state`, `model` and `questions` to `<base_url>/v1/systemone`;
    OpenRouter takes the same body, the same bearer key, at its own path. So the provider is
    given the endpoint's origin as `base_url`, and this rewrites the path on the way out.
    """
    target = httpx2.URL(endpoint)

    async def send_to_endpoint(request: httpx2.Request) -> None:
        if request.url.path == SYSTEM_ONE_PATH:
            request.url = request.url.copy_with(path=target.path)

    return httpx2.AsyncClient(
        event_hooks={"request": [send_to_endpoint]}, transport=transport
    )


def build_jev_model(
    config: ModelConfig, transport: httpx2.AsyncBaseTransport | None = None
) -> Model:
    """The TypeSafe model that config names, with its key given to the provider, not the env.

    `transport` replaces the network, for a test; it applies to a config with an `endpoint`.
    """
    api_key = config.api_key.get_secret_value()
    if config.endpoint is None:
        provider = TypeSafeProvider(api_key=api_key, base_url=config.base_url)
    else:
        origin = httpx2.URL(config.endpoint).copy_with(path="/", query=None)
        provider = TypeSafeProvider(
            api_key=api_key,
            base_url=str(origin).rstrip("/"),
            http_client=build_endpoint_client(config.endpoint, transport),
        )
    return TypeSafeModel(config.model, provider=provider)


async def predict_with_jev(
    config: ModelConfig,
    model: Model,
    sample_id: str,
    label: str,
    jev_input: JevInput,
) -> Prediction:
    """What JEV picks for that input. `config` names the model in events; `label` is copied through."""
    ((field, question),) = jev_input.output_type.model_fields.items()
    sent = {
        "state": jev_input.state,
        "question": question.description,
        "options": list(jev_input.option_to_label),
    }
    started = time.perf_counter()
    try:
        result = await Agent(model, output_type=jev_input.output_type).run(
            jev_input.state
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
    latency_ms = (time.perf_counter() - started) * 1000
    picked = getattr(result.output, field)
    details = result.response.provider_details or {}
    return Prediction(
        id=sample_id,
        label=label,
        input=sent,
        prediction=jev_input.option_to_label.get(picked),
        latency_ms=latency_ms,
        raw_answer=str(picked),
        served_model=result.response.model_name,
        probabilities=details.get("probabilities", {}).get(field),
        confidence=details.get("confidence", {}).get(field),
    )
