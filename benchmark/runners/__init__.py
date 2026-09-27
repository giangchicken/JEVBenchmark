"""facade · one sample through one model: an OpenAI-compatible LLM or JEV."""

from .jev import build_jev_model, predict_with_jev
from .llm import predict_with_llm

__all__ = ["build_jev_model", "predict_with_jev", "predict_with_llm"]
