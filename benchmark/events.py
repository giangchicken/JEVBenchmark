"""adapter · what the run says happened, as one JSON object per line on stdout (`H-6`).

A module logs a record -- the message names the event, `extra=` carries its fields -- and this
turns it into a line. It is installed on `benchmark` and on `agent_toolkit`, whose retry
warnings are worth seeing. No event carries a config, only model names, so no key is logged.
"""

import json
import logging
import sys

TREES = ("benchmark", "agent_toolkit")

# What a LogRecord always carries. Anything else on it came from `extra=` and is the event's own.
BUILT_IN = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonLines(logging.Handler):
    """One event per line. `sys.stdout` is read at emit, so a captured stream is the one used."""

    def emit(self, record: logging.LogRecord) -> None:
        event = {"event": record.getMessage(), "level": record.levelname}
        event.update(
            {
                name: value
                for name, value in record.__dict__.items()
                if name not in BUILT_IN
            }
        )
        print(
            json.dumps(event, ensure_ascii=False, default=str),
            file=sys.stdout,
            flush=True,
        )


def install_structured_events(level: int = logging.INFO) -> None:
    """One `JsonLines` on each tree, replacing one already there."""
    handler = JsonLines()
    for tree in TREES:
        logger = logging.getLogger(tree)
        for installed in [one for one in logger.handlers if isinstance(one, JsonLines)]:
            logger.removeHandler(installed)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
