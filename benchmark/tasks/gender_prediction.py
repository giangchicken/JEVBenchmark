"""logic · the customer's gender, as the conversation shows it, asked of an LLM and of JEV.

The definitions are `gender/gender_prediction.txt`, one line per label. The LLM answers
`{"label": ...}`; JEV picks from `GenderLabel` under the field `gender`.
"""

from typing import get_args

from ..schema import GenderLabel, JevInput, UserLastConversationInput
from .common import (
    PROMPTS,
    build_answer_type,
    build_classification_messages,
    parse_json_label,
    read_label_definition,
    render_conversation,
)

KEY = "label"
LABELS: tuple[str, ...] = get_args(GenderLabel)
DEFINITION = read_label_definition(PROMPTS / "gender" / "gender_prediction.txt")
ANSWER_TYPE = build_answer_type("GenderAnswer", "gender", LABELS, DEFINITION)


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
