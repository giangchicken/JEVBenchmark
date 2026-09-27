"""logic · the customer's sentiment at their last turn, read in context, asked of an LLM and of JEV.

The definitions are `sentiment/sentiment_prediction.txt`, one line per label, which the LLM
gets and answers as `{"label": ...}`. JEV picks from `SentimentLabel` under the field
`sentiment`, asked from `sentiment/sentiment_jev_prediction.json`.
"""

from typing import get_args

from ..schema import JevInput, SentimentLabel, UserLastConversationInput
from .common import (
    PROMPTS,
    build_answer_type,
    build_classification_messages,
    parse_json_label,
    read_jev_definition,
    read_label_definition,
    render_conversation,
)

KEY = "label"
LABELS: tuple[str, ...] = get_args(SentimentLabel)
DEFINITION = read_label_definition(PROMPTS / "sentiment" / "sentiment_prediction.txt")
JEV_DEFINITION = read_jev_definition(
    PROMPTS / "sentiment" / "sentiment_jev_prediction.json", LABELS
)
ANSWER_TYPE = build_answer_type(
    "SentimentAnswer", "sentiment", JEV_DEFINITION.criteria, JEV_DEFINITION.question
)


def build_llm_messages(
    sample_id: str,
    sample_input: UserLastConversationInput,
) -> list[dict[str, object]]:
    return build_classification_messages(DEFINITION, KEY, sample_input.messages)


def parse_llm_answer(
    sample_id: str, sample_input: UserLastConversationInput, answer: str
) -> str | None:
    return parse_json_label(answer, KEY, LABELS)


def build_jev_input(
    sample_id: str, sample_input: UserLastConversationInput
) -> JevInput:
    return JevInput(
        state=render_conversation(sample_input.messages),
        output_type=ANSWER_TYPE,
        option_to_label={label: label for label in LABELS},
    )
