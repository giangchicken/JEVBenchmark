"""logic · the customer's intent at their last turn of a debt-collection call.

The definitions are production's own prompt, `data/intent_prompt.txt`, which speaks in codes
(`6.2`, `13`); `INTENT_CODES` turns a code into the set's label.

- The LLM gets that prompt as one `user` message, filled by `slot_filling` as production fills
  it. The amount the customer promised is not extracted here, so `willpay_number` says so.
  Its answer starts with the code: `"6.2 Đã thanh toán < today"`, `"13. FAQ"`.
- JEV picks one of `IntentLabel` itself, under a field whose description is the prompt's
  INTENT LIST, IMPORTANT RULES and NUMBER NORMALIZATION with every slot written out in words,
  less the rules on how to write the answer (`ANSWER_FORMAT_RULES`) since JEV writes none,
  then which code each label is. The date and the amount are facts about this call, so they go
  in the state.
"""

import re
from pathlib import Path
from typing import get_args

from agent_toolkit.string_utils import slot_filling

from ..schema import INTENT_CODES, IntentInput, IntentLabel, JevInput
from .common import (
    build_answer_type,
    describe_label_codes,
    read_label_definition,
    render_conversation,
)

PROMPT = read_label_definition(
    Path(__file__).resolve().parents[2] / "data" / "intent_prompt.txt"
)
UNKNOWN_PROMISE = "không rõ"

# The question JEV is asked: the prompt from its intent list up to the call's own context.
QUESTION_START = "# INTENT LIST"
QUESTION_END = "# CONTEXT INFORMATION"
SLOTS_IN_WORDS = {
    "sum_OVD_amount": "số nợ quá hạn",
    "last_user_utterance": "câu nói cuối của khách",
    "willpay_number": "số tiền khách hứa trả",
}
# The rules that tell an LLM how to write its answer, by how their line starts.
ANSWER_FORMAT_RULES = ("- Brief response.", "- Do NOT ask questions.")

LABELS: tuple[str, ...] = get_args(IntentLabel)
OPTION_TO_LABEL: dict[str | None, str] = {label: label for label in LABELS}

LEADING_CODE = re.compile(r"^\W*(\d+(?:\.\d+)?)")


def build_intent_question(prompt: str) -> str:
    """The intent list, the rules and the number normalization, with no slot left to fill and no
    answer-format rule, then which code each label is.

    `ValueError` when a section or an answer-format rule is not in the prompt: a prompt that
    changed would otherwise send JEV the rule it was meant to lose.
    """
    start = prompt.index(QUESTION_START)
    end = prompt.index(QUESTION_END, start)
    lines = prompt[start:end].strip().splitlines()
    kept = [line for line in lines if not line.startswith(ANSWER_FORMAT_RULES)]
    if len(lines) - len(kept) != len(ANSWER_FORMAT_RULES):
        raise ValueError(f"the prompt does not hold each of {ANSWER_FORMAT_RULES} once")
    sections = slot_filling("\n".join(kept), SLOTS_IN_WORDS)
    return f"{sections}\n\n{describe_label_codes(INTENT_CODES)}"


ANSWER_TYPE = build_answer_type(
    "IntentAnswer", "intent", LABELS, build_intent_question(PROMPT)
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
