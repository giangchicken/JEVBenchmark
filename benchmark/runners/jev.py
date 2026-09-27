"""adapter · one sample through JEV, as a `Prediction`.

JEV answers a typed question about a text: the task hands in a `JevInput` whose output type has
exactly one field, whose description is the question, and whose state is the text. The pick is
turned into the set's label through `option_to_label`; JEV's distribution and confidence for that
field come from `provider_details`, and the version that answered from the response, because
`jev-latest` is an alias that moves.

`Prediction.input` is what the call was handed and what it put on the wire: `agent`, what
`Agent(model, output_type=...).run(user_prompt)` is given -- the model's name, the output type as
the choice it asks (`read_choice_question`: its name, field, question, each option's criterion,
what picking none means), the user prompt -- and `request`, the last request the JEV client sent for it
(`method`, `url`, `body`), or `None` when none left. The SDK's retries resend the same body, so
the last one stands for all. Headers are left out, because one of them carries the key.

A failed call never raises: it is a `Prediction` with `error` set and one `prediction_failed` event;
a pick `option_to_label` does not know is a parse error, as it would be for an LLM.
"""

import json
import time
from contextvars import ContextVar
from typing import Any

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
from ..tasks.common import read_choice_question

logger = get_logger(__name__)

# A first run prints a banner to stdout, which is where the events go.
pydantic_ai.BANNER_ENABLED = False

# Where the JEV client puts the requests it sends for the sample being asked. A context variable
# and not a list on the client: one client serves every sample of a run at once, each in its own
# task, and a task sees only the list its own `predict_with_jev` set.
SENT_REQUESTS: ContextVar[list[httpx2.Request] | None] = ContextVar(
    "sent_requests", default=None
)


async def record_request(request: httpx2.Request) -> None:
    """A request hook: add the request to the calling sample's list, when one is set."""
    recorded = SENT_REQUESTS.get()
    if recorded is not None:
        recorded.append(request)


def build_jev_client(
    endpoint: str | None, transport: httpx2.AsyncBaseTransport | None = None
) -> httpx2.AsyncClient:
    """The client JEV's provider sends through: it records every request it sends, and, given
    an endpoint, sends TypeSafe's decisions call to that URL instead.

    The TypeSafe SDK always posts `state`, `model` and `questions` to `<base_url>/v1/systemone`;
    OpenRouter takes the same body, the same bearer key, at its own path. So the provider is
    given the endpoint's origin as `base_url`, and this rewrites the path on the way out --
    before the request is recorded, so the record says where it went.
    """
    hooks = [record_request]
    if endpoint is not None:
        target = httpx2.URL(endpoint)

        async def send_to_endpoint(request: httpx2.Request) -> None:
            if request.url.path == SYSTEM_ONE_PATH:
                request.url = request.url.copy_with(path=target.path)

        hooks.insert(0, send_to_endpoint)
    return httpx2.AsyncClient(event_hooks={"request": hooks}, transport=transport)


def build_jev_model(
    config: ModelConfig, transport: httpx2.AsyncBaseTransport | None = None
) -> Model:
    """The TypeSafe model that config names, with its key given to the provider, not the env.

    `transport` replaces the network, for a test.
    """
    api_key = config.api_key.get_secret_value()
    client = build_jev_client(config.endpoint, transport)
    if config.endpoint is None:
        base_url = config.base_url
    else:
        origin = httpx2.URL(config.endpoint).copy_with(path="/", query=None)
        base_url = str(origin).rstrip("/")
    provider = TypeSafeProvider(api_key=api_key, base_url=base_url, http_client=client)
    return TypeSafeModel(config.model, provider=provider)


def build_sent_input(
    model: Model, jev_input: JevInput, recorded: list[httpx2.Request]
) -> dict[str, Any]:
    """`Prediction.input`: what `Agent` and `.run` were handed, and the last request sent."""
    last = recorded[-1] if recorded else None
    return {
        "agent": {
            "model": model.model_name,
            "output_type": read_choice_question(jev_input.output_type).model_dump(),
            "user_prompt": jev_input.state,
        },
        "request": None
        if last is None
        else {
            "method": last.method,
            "url": str(last.url),
            "body": json.loads(last.content),
        },
    }


async def predict_with_jev(
    config: ModelConfig,
    model: Model,
    sample_id: str,
    label: str,
    jev_input: JevInput,
) -> Prediction:
    """What JEV picks for that input. `config` names the model in events; `label` is copied through."""
    ((field, _),) = jev_input.output_type.model_fields.items()
    recorded: list[httpx2.Request] = []
    token = SENT_REQUESTS.set(recorded)
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
            input=build_sent_input(model, jev_input, recorded),
            prediction=None,
            error=message,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    finally:
        SENT_REQUESTS.reset(token)
    latency_ms = (time.perf_counter() - started) * 1000
    picked = getattr(result.output, field)
    details = result.response.provider_details or {}
    return Prediction(
        id=sample_id,
        label=label,
        input=build_sent_input(model, jev_input, recorded),
        prediction=jev_input.option_to_label.get(picked),
        latency_ms=latency_ms,
        raw_answer=str(picked),
        served_model=result.response.model_name,
        probabilities=details.get("probabilities", {}).get(field),
        confidence=details.get("confidence", {}).get(field),
    )
