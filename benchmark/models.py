"""adapter · `config/model/*.json` into the models a run asks.

One file is one model, named by its stem: `gemma4-31B-it.json` is `gemma4-31B-it`, whatever its
`model` key says, because the stem is what `--models` and the report use. A file whose
`provider` is `"typesafe"` is JEV; any other is an OpenAI-compatible LLM.

Every file is read and checked before anything is asked, so a broken one stops the run with
`ConfigError` naming that file and not halfway through a dataset. The key is held as a
`SecretStr`, so a config printed, logged or dumped reads `**********`: the key is only ever
unwrapped where it is handed to a client, and nothing logs it.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "model"
JEV_PROVIDER = "typesafe"
# What `extra` may not pass to `complete`: the config's own keys, which are passed already.
RESERVED_EXTRA = frozenset(
    {"model", "api_key", "base_url", "messages", "prompt", "enable_thinking"}
    | {"temperature", "top_p"}
)


class ConfigError(Exception):
    """A model config is missing, broken or incomplete, and nothing has been asked yet."""


class ModelConfig(BaseModel):
    """One model file. `extra="forbid"`: a misspelt key is an error, not a silent default."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str  # the file stem
    provider: Literal["openai", "typesafe"] = "openai"
    model: str = Field(min_length=1)
    api_key: SecretStr
    base_url: str | None = None  # required for an LLM; JEV defaults to TypeSafe's own
    # JEV only: the full URL of the decisions call when it is not TypeSafe's own -- OpenRouter's
    # is `https://openrouter.ai/api/alpha/decisions`. Takes the place of `base_url`.
    endpoint: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_concurrency: int = Field(default=10, gt=0)  # requests in flight to this model
    enable_reason: bool | None = None  # sent as `enable_thinking`
    extra: dict[str, Any] = Field(
        default_factory=dict
    )  # to `complete`, e.g. `max_tokens`

    @model_validator(mode="after")
    def check_endpoint(self) -> Self:
        if not self.api_key.get_secret_value():
            raise ValueError("api_key is empty")
        if not self.is_jev and not self.base_url:
            raise ValueError("an LLM needs a base_url")
        if self.endpoint is not None and (not self.is_jev or self.base_url):
            raise ValueError("endpoint is for JEV only, and replaces base_url")
        reserved = sorted(RESERVED_EXTRA & self.extra.keys())
        if reserved:
            raise ValueError(
                f"extra may not set {reserved}: they are the config's own keys"
            )
        return self

    @property
    def is_jev(self) -> bool:
        return self.provider == JEV_PROVIDER


def load_model_config(path: Path) -> ModelConfig:
    """The model that file declares, or `ConfigError` naming the file and what is wrong."""
    try:
        declared = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{path.name}: cannot read it as JSON: {error}") from error
    if not isinstance(declared, dict):
        raise ConfigError(f"{path.name}: must be a JSON object")
    try:
        return ModelConfig.model_validate({**declared, "name": path.stem})
    except ValidationError as error:
        # `include_input=False`: the input is the file itself, key included.
        problems = "; ".join(
            f"{'.'.join(map(str, problem['loc'])) or 'file'}: {problem['msg']}"
            for problem in error.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"{path.name}: {problems}") from error


def load_model_configs(
    directory: Path = CONFIG_DIR, names: Iterable[str] | None = None
) -> tuple[ModelConfig, ...]:
    """Every model in that directory, sorted by name, or only the named ones.

    `ConfigError` when the directory holds no model or a name has no file -- a run that asked
    for a model and quietly left it out would report a comparison nobody asked for.
    """
    paths = {path.stem: path for path in sorted(directory.glob("*.json"))}
    wanted = list(paths) if names is None else list(dict.fromkeys(names))
    missing = [name for name in wanted if name not in paths]
    if missing:
        raise ConfigError(
            f"no config for {missing} in {directory}; there is {sorted(paths)}"
        )
    if not wanted:
        raise ConfigError(f"no model config in {directory}")
    return tuple(load_model_config(paths[name]) for name in wanted)
