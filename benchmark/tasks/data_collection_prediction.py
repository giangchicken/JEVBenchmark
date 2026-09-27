"""logic · the call outcome (`action_code`) collected from a whole call, fec's or vpb's.

One task over two sources of the same shape. The id's prefix says which source a sample is
from, and each source brings its own definitions -- its `action_code` description, verbatim,
in `data_collection/<source>_prediction.txt` -- and its own answers:

- fec's are its labels themselves;
- vpb's are codes (`"205"`, `"1107"`, `"N/A"`), which `VPB_CODES` turns into the set's labels.

An LLM's answer is read from `{"action_code": ...}` or, failing that, from the one answer
written alone -- `1` is not found inside `1107`.

JEV picks the source's labels themselves, asked from `data_collection/<source>_jev_prediction.json`,
one criterion per label.
"""

from dataclasses import dataclass
from typing import get_args

from pydantic import BaseModel

from ..schema import VPB_CODES, ConversationInput, FecLabel, JevInput, read_source
from .common import (
    PROMPTS,
    build_answer_type,
    build_classification_messages,
    parse_code_answer,
    read_jev_definition,
    read_label_definition,
    render_conversation,
)

KEY = "action_code"


@dataclass(frozen=True)
class Source:
    """What one source asks: its definitions, an LLM's answers each mapped to a label, and the
    answer type JEV picks the source's labels from."""

    definition: str
    answer_to_label: dict[str, str]
    answer_type: type[BaseModel]


def build_source(source: str, answer_to_label: dict[str, str]) -> Source:
    """A source whose LLM answers are those keys; JEV is asked for the labels themselves."""
    definition = read_label_definition(
        PROMPTS / "data_collection" / f"{source}_prediction.txt"
    )
    asked = read_jev_definition(
        PROMPTS / "data_collection" / f"{source}_jev_prediction.json",
        list(answer_to_label.values()),
    )
    answer_type = build_answer_type(
        f"{source.capitalize()}Answer", KEY, asked.criteria, asked.question
    )
    return Source(definition, answer_to_label, answer_type)


SOURCES = {
    "fec": build_source("fec", {label: label for label in get_args(FecLabel)}),
    "vpb": build_source("vpb", dict(VPB_CODES)),
}


def build_llm_messages(
    sample_id: str, sample_input: ConversationInput
) -> list[dict[str, object]]:
    definition = SOURCES[read_source(sample_id)].definition
    return build_classification_messages(definition, KEY, sample_input.messages)


def parse_llm_answer(
    sample_id: str, sample_input: ConversationInput, answer: str
) -> str | None:
    return parse_code_answer(
        answer, KEY, SOURCES[read_source(sample_id)].answer_to_label
    )


def build_jev_input(sample_id: str, sample_input: ConversationInput) -> JevInput:
    source = SOURCES[read_source(sample_id)]
    option_to_label: dict[str | None, str] = {
        label: label for label in source.answer_to_label.values()
    }
    return JevInput(
        state=render_conversation(sample_input.messages),
        output_type=source.answer_type,
        option_to_label=option_to_label,
    )
