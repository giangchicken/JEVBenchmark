"""facade · one task per dataset, each a module offering what `schema.Task` declares."""

from . import (
    data_collection_prediction,
    gender_prediction,
    intent_prediction,
    sentiment_prediction,
    tool_decision_prediction,
)

__all__ = [
    "data_collection_prediction",
    "gender_prediction",
    "intent_prediction",
    "sentiment_prediction",
    "tool_decision_prediction",
]
