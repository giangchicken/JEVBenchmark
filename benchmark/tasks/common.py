"""logic · what every dataset's task does the same way: render, ask, read an answer back.

A task turns one `sample.input` into two questions -- chat messages for an LLM and a `JevInput`
for JEV -- and each set has its label definitions in `config/prompts/benchmark/<dataset>/`,
one `*_prediction.txt` per question, that both are built from: the LLM prompt carries it
verbatim and JEV's field description is it.
That file is read once, when the task module is imported.
"""

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from agent_toolkit.string_utils import extract_json_from_text
from pydantic import BaseModel, Field, create_model

from ..schema import ToolSpec, Turn

PROMPTS = Path(__file__).resolve().parents[2] / "config" / "prompts" / "benchmark"


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


def build_answer_type(
    name: str,
    field: str,
    options: Sequence[str],
    description: str,
    allow_none: bool = False,
) -> type[BaseModel]:
    """A one-field model asking JEV to pick one of those options -- or none, when allowed.

    The description is the question: JEV reads it, and it is not part of the state.
    """
    choice: Any = cast(Any, Literal)[tuple(options)]
    if allow_none:
        choice = choice | None
    fields: dict[str, Any] = {field: (choice, Field(description=description))}
    return create_model(name, **fields)


def describe_label_codes(codes: Mapping[str, str]) -> str:
    """One line per label saying which code of the definitions it is, for a question asked by
    label: JEV picks the set's labels, while the definitions above them speak in codes."""
    lines = [f"- {label}: {code}" for code, label in codes.items()]
    return "Mỗi lựa chọn là tên của một mã trong định nghĩa ở trên:\n" + "\n".join(
        lines
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
