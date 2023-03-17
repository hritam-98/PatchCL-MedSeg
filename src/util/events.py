from collections import defaultdict
from enum import Enum, auto
from typing import Any, Callable


class EventTypes(Enum):
    END_OF_TRAINING_BATCH = auto()
    END_OF_VALIDATION_BATCH = auto()


event_handlers: dict[EventTypes, list[Callable[[Any], Any]]] = defaultdict(list)


def register_event_handler(event_type: EventTypes, handler: Callable[[Any], Any]) -> None:
    event_handlers[event_type].append(handler)


def post_event(event_type: EventTypes, **data):
    for handler in event_handlers.get(event_type, []):
        handler(**data)
