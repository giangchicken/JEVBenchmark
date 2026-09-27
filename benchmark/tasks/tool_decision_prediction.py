"""logic · which tool the assistant should call next, or none, from the sample's own catalog.

The definitions are the corpus's own system prompt, verbatim. Only the tool's name is scored.

- The LLM gets that prompt and the catalog as the system message, then the rendered
  conversation as one user message, and answers a JSON array of names: `[]` is `NO_TOOL`, one name of the catalog is that
  tool, anything else is no answer.
- JEV picks a name of the catalog or none, asked `JEV_QUESTION` followed by the catalog: the
  corpus prompt asks for a JSON array of any length, which a single pick cannot answer. The
  catalog is how each option is defined, so it is part of the question, and the state is the
  conversation alone. The options differ per sample, so the answer type is built per sample.
"""

import json
from typing import Any

from agent_toolkit.string_utils import extract_json_from_text

from ..schema import NO_TOOL, JevInput, ToolDecisionInput
from .common import (
    PROMPTS,
    build_answer_type,
    read_label_definition,
    render_conversation,
    render_tools,
)

DEFINITION = read_label_definition(PROMPTS / "tool_decision" / "tool_prediction.txt")
ANSWER_FORMAT = (
    "You do not reply to the customer. Output only the JSON array of tool names,"
    ' for example [] or ["TOOL_NAME"].'
)
JEV_QUESTION = (
    "Based on the available tools and conversation history, which tool should the assistant"
    " call next? Pick none if no tool should be called."
)


def build_llm_messages(
    sample_id: str, sample_input: ToolDecisionInput
) -> list[dict[str, object]]:
    """The definitions, the catalog and the answer format as system; the conversation as user.

    The conversation is rendered into one message rather than replayed as chat turns: replayed,
    the model takes the assistant's seat and answers the customer instead of routing.
    """
    system = f"{DEFINITION}\n\n# Tools\n{render_tools(sample_input.tools)}\n\n{ANSWER_FORMAT}"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Conversation:\n{render_conversation(sample_input.messages)}",
        },
    ]


def read_json_array(answer: str) -> list[Any] | None:
    """The array the answer holds, or `None`.

    The span from the first `[` to the last `]` is tried as JSON first: `extract_json_from_text`
    looks for an object before an array, so an array of call objects comes back as its first
    object.
    """
    start, end = answer.find("["), answer.rfind("]")
    if 0 <= start < end:
        try:
            found = json.loads(answer[start : end + 1])
        except json.JSONDecodeError:
            found = None
        if isinstance(found, list):
            return found
    found = extract_json_from_text(answer)
    return found if isinstance(found, list) else None


def parse_llm_answer(
    sample_id: str, sample_input: ToolDecisionInput, answer: str
) -> str | None:
    """`NO_TOOL` for `[]`, the name for one catalog name (repeated or not), else `None`.

    An element may be the name itself or a call object carrying it, `{"name": ...}`: the
    arguments are not scored, so the two say the same thing.
    """
    found = read_json_array(answer)
    if found is None:
        return None
    if not found:
        return NO_TOOL
    names = {tool.name for tool in sample_input.tools}
    called = {
        str(element.get("name")) if isinstance(element, dict) else str(element)
        for element in found
    }
    return called.pop() if len(called) == 1 and called <= names else None


def build_jev_input(sample_id: str, sample_input: ToolDecisionInput) -> JevInput:
    names = [tool.name for tool in sample_input.tools]
    question = f"{JEV_QUESTION}\n\nTools:\n{render_tools(sample_input.tools)}"
    option_to_label: dict[str | None, str] = {name: name for name in names}
    option_to_label[None] = NO_TOOL
    return JevInput(
        state=render_conversation(sample_input.messages),
        output_type=build_answer_type(
            "ToolAnswer", "tool", names, question, allow_none=True
        ),
        option_to_label=option_to_label,
    )
