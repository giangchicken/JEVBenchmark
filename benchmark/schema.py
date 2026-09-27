"""shape · the five test sets, one sample type and one file type each, and the JEV input a
task builds from a sample.

Every sample is `{id, input, label}` and nothing else. A set is a JSON array of its samples,
loaded with `<Name>TestSet.model_validate_json`. Invariants every set holds:

- ids are unique within the set;
- intent, tool_decision, gender and sentiment end on a `user` turn, because the label is a
  judgement about what the customer just said; data_collection holds whole calls, which may
  end on either role;
- a data_collection id starts with its source, `fec-` or `vpb-`, and its label is one of that
  source's;
- a tool_decision label is one of that sample's own tool names, or `NO_TOOL`.

What a benchmark run makes of a sample is a `Prediction`, one per model per sample, and what a
dataset module offers the run is a `Task`.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Generic, Literal, Protocol, Self, TypeVar, get_args

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

NO_TOOL: Final = "None"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Turn(Frozen):
    role: Literal["assistant", "user"]
    content: str = Field(min_length=1)


class ConversationInput(Frozen):
    messages: tuple[Turn, ...] = Field(min_length=1)


class UserLastConversationInput(ConversationInput):
    """A conversation cut right after the customer spoke: the last turn is `user`."""

    @model_validator(mode="after")
    def check_last_turn_is_user(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("the last turn must be the user's")
        return self


class Sample(Frozen):
    """What every sample has besides its own `input` and `label`."""

    id: str


S = TypeVar("S", bound=Sample)


class TestSet(RootModel[tuple[S, ...]], Generic[S]):
    """One file: a JSON array of samples whose ids are unique."""

    @model_validator(mode="after")
    def check_unique_ids(self) -> Self:
        counts = Counter(sample.id for sample in self.root)
        repeated = sorted(id_ for id_, count in counts.items() if count > 1)
        if repeated:
            raise ValueError(f"duplicate ids: {repeated}")
        return self


# ---------- data_collection: fec ----------
FecLabel = Literal[
    "Non_Qualified_Customer",
    "DNC_request",
    "To_Follow_Up",
    "No_Demand",
    "Success_A_XS",
    "Success_B_XS",
    "Other",
]


# ---------- data_collection: vpb ----------
# A name per outcome, for reading results. The source and its prompt speak in codes, and
# `VPB_CODES` is the one place the two meet.
VpbLabel = Literal[
    "THIRD_PARTY_RELATIVE",
    "THIRD_PARTY_COMPLAINT",
    "THIRD_PARTY_PROMISE",
    "WRONG_NUMBER",
    "CALL_DROPPED",
    "NO_ANSWER",
    "UNREACHABLE",
    "PROMISE_TO_PAY",
    "REFUSE_OR_DELAY",
    "ALREADY_PAID",
    "COMPLAINT_OR_INQUIRY",
    "UNDETERMINED",
]

VPB_CODES: Final[dict[str, VpbLabel]] = {
    "205": "THIRD_PARTY_RELATIVE",  # not the customer; a relative; no complaint, no promise
    "303": "THIRD_PARTY_COMPLAINT",  # not the customer; complains
    "1107": "THIRD_PARTY_PROMISE",  # not the customer; promises to pay on their behalf
    "206": "WRONG_NUMBER",
    "207": "CALL_DROPPED",  # cut off, or stopped talking after confirming who they are
    "208": "NO_ANSWER",
    "209": "UNREACHABLE",
    "1": "PROMISE_TO_PAY",  # the customer; a serious promise with a concrete time
    "202": "REFUSE_OR_DELAY",  # the customer; refuses, evades, or promises vaguely
    "203": "ALREADY_PAID",
    "204": "COMPLAINT_OR_INQUIRY",  # the customer; complains or asks for information
    "N/A": "UNDETERMINED",
}


# ---------- data_collection ----------
# One task over two sources of the same shape -- a call, and the outcome field to collect from
# it -- told apart by the id's prefix, because each source has its own definitions and labels.
DATA_COLLECTION_LABELS: Final[dict[str, tuple[str, ...]]] = {
    "fec": get_args(FecLabel),
    "vpb": get_args(VpbLabel),
}


def read_source(sample_id: str) -> str:
    """The source an id's prefix names: `fec-…` is `fec`. `ValueError` for any other prefix."""
    source = sample_id.split("-", 1)[0]
    if source not in DATA_COLLECTION_LABELS or not sample_id.startswith(f"{source}-"):
        raise ValueError(
            f"id {sample_id!r} does not start with a source: {list(DATA_COLLECTION_LABELS)}"
        )
    return source


class DataCollectionSample(Sample):
    input: ConversationInput
    label: FecLabel | VpbLabel

    @model_validator(mode="after")
    def check_label_of_source(self) -> Self:
        source = read_source(self.id)
        if self.label not in DATA_COLLECTION_LABELS[source]:
            raise ValueError(f"label {self.label!r} is not a {source} label")
        return self


DataCollectionTestSet = TestSet[DataCollectionSample]


# ---------- intent ----------
# A name per intent, for reading results. `data/intent_prompt.txt` speaks in codes, and
# `INTENT_CODES` is the one place the two meet.
IntentLabel = Literal[
    "AGREE_TO_PAY",
    "REFUSE_TO_PAY",
    "PROMISE_FULL_AMOUNT",
    "PROMISE_PARTIAL_AMOUNT",
    "PAID_NO_DATE",
    "PAID_BEFORE_TODAY",
    "PAID_TODAY",
    "BUSY",
    "ASK_REPEAT",
    "FALLBACK",
    "THIRD_PARTY",
    "WANTS_TO_SPEAK",
    "FAQ",
    "VIRTUAL_ASSISTANT",
]

INTENT_CODES: Final[dict[str, IntentLabel]] = {
    "1": "AGREE_TO_PAY",
    "3": "REFUSE_TO_PAY",
    "4": "PROMISE_FULL_AMOUNT",  # PTP >= the overdue amount
    "5": "PROMISE_PARTIAL_AMOUNT",  # PTP < the overdue amount
    "6.1": "PAID_NO_DATE",
    "6.2": "PAID_BEFORE_TODAY",  # yesterday, the day before, last week
    "6.3": "PAID_TODAY",
    "7": "BUSY",
    "8": "ASK_REPEAT",
    "9": "FALLBACK",
    "11": "THIRD_PARTY",
    "12": "WANTS_TO_SPEAK",  # "em ơi chị bảo này": about to say or ask something
    "13": "FAQ",
    "20": "VIRTUAL_ASSISTANT",
}


class IntentInput(UserLastConversationInput):
    # Strict: a JSON `0` is not 1970-01-01 and `true` is not one dong. ISO strings still load.
    today: date = Field(strict=True)
    sum_OVD_amount: int = Field(strict=True, gt=0, description="VND")


class IntentSample(Sample):
    input: IntentInput
    label: IntentLabel


IntentTestSet = TestSet[IntentSample]


# ---------- tool_decision ----------
class ToolSpec(Frozen):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDecisionInput(UserLastConversationInput):
    tools: tuple[ToolSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_tool_names(self) -> Self:
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        if NO_TOOL in names:
            raise ValueError(
                f"a tool may not be named {NO_TOOL!r}: it is the no-call label"
            )
        return self


class ToolDecisionSample(Sample):
    input: ToolDecisionInput
    label: str  # a name in `input.tools`, or `NO_TOOL`

    @model_validator(mode="after")
    def check_label_in_catalog(self) -> Self:
        names = {tool.name for tool in self.input.tools}
        if self.label != NO_TOOL and self.label not in names:
            raise ValueError(f"label {self.label!r} is neither a tool nor {NO_TOOL!r}")
        return self


ToolDecisionTestSet = TestSet[ToolDecisionSample]


# ---------- gender ----------
GenderLabel = Literal["male", "female", "unknown"]


class GenderSample(Sample):
    input: UserLastConversationInput
    label: GenderLabel


GenderTestSet = TestSet[GenderSample]


# ---------- sentiment ----------
SentimentLabel = Literal["positive", "neutral", "negative"]


class SentimentSample(Sample):
    input: UserLastConversationInput
    label: SentimentLabel


SentimentTestSet = TestSet[SentimentSample]


# ---------- JEV input ----------
@dataclass(frozen=True)
class JevInput:
    """What JEV judges: a state and a one-field output type, both built from `sample.input`."""

    state: str  # the content to judge
    output_type: type[
        BaseModel
    ]  # one field; the question is its `Field(description=...)`
    option_to_label: dict[str | None, str]  # JEV's option -> the set's label


# ---------- a benchmark run ----------
class Task(Protocol):
    """What one dataset's module offers a run: the same `sample.input` for an LLM and for JEV.

    The id goes along because in data_collection its prefix says which source's definitions
    apply; the label never does.
    """

    def build_llm_messages(
        self, sample_id: str, sample_input: Any
    ) -> list[dict[str, object]]: ...

    def parse_llm_answer(
        self, sample_id: str, sample_input: Any, answer: str
    ) -> str | None: ...

    def build_jev_input(self, sample_id: str, sample_input: Any) -> JevInput: ...


class Prediction(Frozen):
    """One model's answer to one sample, as a line of `predictions/<model>.jsonl`.

    Exactly one of three outcomes, told apart by two fields:
    - answered: `prediction` is a label of the set, `error` is `None`;
    - unreadable answer (a parse error): both are `None`, `raw_answer` holds what came back;
    - failed call: `prediction` is `None` and `error` is `"<Type>: <message>"`.
    Both failures count as wrong, so every model is scored on the same n.

    `input` is what the model was sent, failed call or not: an LLM's `{"complete": {...}}`, JEV's
    `{"agent": {...}, "request": {...}}` (the runners say what each holds). It is written with
    the prediction and again, on its own, to `input/<model>.jsonl`. A run made before 2026-09-27
    holds the older `{"messages"}` / `{"state", "question", "options"}` here, and JEV's file of
    run `20260927-jev-written-criteria` holds none, which reads back as `{}`.
    """

    id: str
    label: str
    input: dict[str, Any] = Field(default_factory=dict)
    prediction: str | None
    error: str | None = None
    latency_ms: float = Field(ge=0)
    raw_answer: str | None = None
    served_model: str | None = (
        None  # the version the endpoint says answered: `jev-latest` moves
    )
    probabilities: dict[str, float] | None = None  # JEV only: its option -> probability
    confidence: float | None = None  # JEV only, 0 undecided to 1

    @property
    def correct(self) -> bool:
        return self.prediction == self.label
