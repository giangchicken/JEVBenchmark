"""logic · the customer's intent at their last turn of a debt-collection call.

The definitions are production's own prompt, `data/intent_prompt.txt`, which speaks in codes
(`6.2`, `13`); `INTENT_CODES` turns a code into the set's label.

- The LLM gets that prompt as one `user` message, filled by `slot_filling` as production fills
  it. The amount the customer promised is not extracted here, so `willpay_number` says so.
  Its answer starts with the code: `"6.2 Đã thanh toán < today"`, `"13. FAQ"`.
- JEV picks one of `IntentLabel` itself, asked from `intent/intent_jev_prediction.json`, one
  criterion per label. The date and the amount are facts about this call, so they go in the
  state.
"""

import re
from pathlib import Path
from typing import get_args

from agent_toolkit.string_utils import slot_filling

from ..schema import INTENT_CODES, IntentInput, IntentLabel, JevInput
from .common import (
    PROMPTS,
    build_answer_type,
    read_jev_definition,
    read_label_definition,
    render_conversation,
)

PROMPT = read_label_definition(
    Path(__file__).resolve().parents[2] / "data" / "intent_prompt.txt"
)
UNKNOWN_PROMISE = "không rõ"

LABELS: tuple[str, ...] = get_args(IntentLabel)
OPTION_TO_LABEL: dict[str | None, str] = {label: label for label in LABELS}

LEADING_CODE = re.compile(r"^\W*(\d+(?:\.\d+)?)")

JEV_DEFINITION = read_jev_definition(
    PROMPTS / "intent" / "intent_jev_prediction.json", LABELS
)
ANSWER_TYPE = build_answer_type(
    "IntentAnswer", "intent", JEV_DEFINITION.criteria, JEV_DEFINITION.question
)


def build_llm_messages(
    sample_id: str, sample_input: IntentInput
) -> list[dict[str, object]]:
    slots = {
        "TODAY_PROMPT": sample_input.today.isoformat(),
        "sum_OVD_amount": sample_input.sum_OVD_amount,
        "chat_his": render_conversation(sample_input.messages),
        "last_user_utterance": sample_input.messages[-1].content,
        "willpay_number": UNKNOWN_PROMISE,
    }
    return [{"role": "user", "content": slot_filling(PROMPT, slots)}]


def parse_llm_answer(
    sample_id: str, sample_input: IntentInput, answer: str
) -> str | None:
    """The label for the code the answer starts with; `None` for no code or an unknown one."""
    found = LEADING_CODE.match(answer)
    return INTENT_CODES.get(found[1]) if found else None


def build_jev_input(sample_id: str, sample_input: IntentInput) -> JevInput:
    context = (
        f"hôm nay: {sample_input.today.isoformat()}\n"
        f"số nợ quá hạn: {sample_input.sum_OVD_amount} đồng\n"
    )
    return JevInput(
        state=context + render_conversation(sample_input.messages),
        output_type=ANSWER_TYPE,
        option_to_label=OPTION_TO_LABEL,
    )
