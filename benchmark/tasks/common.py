"""logic · what every dataset's task does the same way: render, ask, read an answer back.

A task turns one `sample.input` into two questions -- chat messages for an LLM and a `JevInput`
for JEV -- and each set has its label definitions in `config/prompts/benchmark/<dataset>/`,
one `*_prediction.txt` per question, which the LLM prompt carries verbatim. JEV is asked from
its own file beside it, `*_jev_prediction.json`, written by hand from that definition: the
`question` says what holds for every option, and `criteria` what each option means on its own.
The two are kept in step by whoever edits one of them; nothing derives one from the other.
Both are read once, when the task module is imported.
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Union, cast

from agent_toolkit.string_utils import extract_json_from_text
from pydantic import BaseModel, Field, TypeAdapter, WithJsonSchema, create_model

from ..schema import Frozen, ToolSpec, Turn

PROMPTS = Path(__file__).resolve().parents[2] / "config" / "prompts" / "benchmark"


class JevDefinition(Frozen):
    """A `*_jev_prediction.json`: what JEV is asked of every option, and each option's criterion,
    in the order JEV is offered them."""

    question: str = Field(min_length=1)
    criteria: dict[str, Annotated[str, Field(min_length=1)]]


def read_label_definition(path: Path) -> str:
    """The definitions in that file. Raises rather than returning `""`, as `read_txt` would:
    a prompt that silently lost its definitions still gets answers, and they mean nothing."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    return text


def render_conversation(messages: Sequence[Turn]) -> str:
    """One line per turn, `role: content`, in order."""
    return "\n".join(f"{turn.role}: {turn.content}" for turn in messages)


def render_tools(tools: Sequence[ToolSpec]) -> str:
    """The catalog as a JSON array of `{name, description, parameters}`, in the sample's order."""
    return json.dumps(
        [tool.model_dump() for tool in tools], ensure_ascii=False, indent=2
    )


def build_classification_messages(
    definition: str, key: str, messages: Sequence[Turn]
) -> list[dict[str, object]]:
    """A system message holding the definitions and the answer format, then the conversation."""
    instruction = (
        f'Chỉ trả về JSON dạng {{"{key}": "<giá trị>"}}, không giải thích thêm.'
    )
    return [
        {"role": "system", "content": f"{definition}\n\n{instruction}"},
        {"role": "user", "content": render_conversation(messages)},
    ]


def read_jev_definition(path: Path, labels: Sequence[str]) -> JevDefinition:
    """That file's question and criteria.

    `ValueError` unless it gives every label of the set a criterion and no other label one: a
    label left out could never be picked, and one the set lacks could be picked but not scored.
    """
    definition = JevDefinition.model_validate_json(path.read_text(encoding="utf-8"))
    if sorted(definition.criteria) != sorted(labels):
        raise ValueError(
            f"{path}: criteria for {sorted(definition.criteria)}, the set has {sorted(labels)}"
        )
    return definition


def build_answer_type(
    name: str,
    field: str,
    criteria: Mapping[str, str],
    question: str,
    none_criterion: str | None = None,
) -> type[BaseModel]:
    """A one-field model asking JEV to pick one of those options, or none when `none_criterion`
    says what none means.

    TypeSafe builds the choice from the field's JSON schema: the field description is its
    `question`, and each option's own description is that option's entry in `criteria`, so an
    option is `Annotated[Literal[option], Field(description=criterion)]`. It reads an optional
    field only as an `anyOf` of exactly two, `null` and the options, while pydantic writes
    `Literal | None` as one flat `anyOf`, which TypeSafe refuses; so with none allowed the
    schema is given explicitly, as the options' own schema nested beside `null`.
    """
    members = tuple(
        Annotated[Literal[option], Field(description=criterion)]
        for option, criterion in criteria.items()
    )
    choice: Any = cast(Any, Union)[members]
    if none_criterion is not None:
        schema = {
            "anyOf": [
                TypeAdapter(choice).json_schema(),
                {"type": "null", "description": none_criterion},
            ]
        }
        choice = Annotated[choice | None, WithJsonSchema(schema)]
    fields: dict[str, Any] = {field: (choice, Field(description=question))}
    return create_model(name, **fields)


class ChoiceQuestion(Frozen):
    """What an answer type asks JEV, in the words `build_answer_type` was given: the type's
    name, its one field, the question, each option's criterion in the order JEV is offered
    them, and what picking none means -- `None` when none is not offered."""

    name: str
    field: str
    question: str
    criteria: dict[str, str | None]  # `None`: that option has no description
    none_criterion: str | None = None


def read_choice_question(output_type: type[BaseModel]) -> ChoiceQuestion:
    """The choice that answer type offers, read off its JSON schema the way TypeSafe reads it:
    each `const` is an option and its `description` the criterion, a `null` beside them is
    picking none (`""` when it has no description), and the options may sit one `anyOf` down."""
    ((field, schema),) = output_type.model_json_schema()["properties"].items()
    criteria: dict[str, str | None] = {}
    none_criterion: str | None = None

    def add(member: dict[str, Any]) -> None:
        nonlocal none_criterion
        if member.get("type") == "null":
            none_criterion = member.get("description", "")
        elif "anyOf" in member:
            for option in member["anyOf"]:
                add(option)
        elif (
            "enum" in member
        ):  # a plain `Literal`: options with no criterion of their own
            criteria.update(dict.fromkeys(member["enum"]))
        else:
            criteria[member["const"]] = member.get("description")

    add(schema)
    return ChoiceQuestion(
        name=output_type.__name__,
        field=field,
        question=schema.get("description", ""),
        criteria=criteria,
        none_criterion=none_criterion,
    )


def parse_json_label(answer: str, key: str, labels: Sequence[str]) -> str | None:
    """The value under `key` in the answer's JSON object when it is one of the labels, else `None`."""
    found = extract_json_from_text(answer)
    if not isinstance(found, dict) or found.get(key) is None:
        return None
    value = str(found[key]).strip()
    return value if value in labels else None


def parse_code_answer(answer: str, key: str, codes: Mapping[str, str]) -> str | None:
    """The label for the code the answer gives, read as JSON first and as a lone token second.

    The fallback accepts a code written on its own anywhere in the answer, as long as exactly
    one code is: `1` does not match inside `1107` nor `205` inside `205.000`, and two
    different codes is no answer.
    """
    code = parse_json_label(answer, key, list(codes))
    if code is None:
        written = {
            candidate
            for candidate in codes
            if re.search(
                rf"(?<![\w/])(?<!\d[.,]){re.escape(candidate)}(?![\w/]|[.,]\d)", answer
            )
        }
        code = written.pop() if len(written) == 1 else None
    return None if code is None else codes[code]
