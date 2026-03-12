from dataclasses import dataclass, field


@dataclass
class StatusEvent:
    content: str = ""
    type: str = "status"


@dataclass
class ResponseStartEvent:
    type: str = "response_start"


@dataclass
class ResponseDelta:
    content: str = ""
    type: str = "response_delta"


@dataclass
class ResponseEndEvent:
    type: str = "response_end"


@dataclass
class CardsEvent:
    content: list[dict] = field(default_factory=list)
    type: str = "cards"


@dataclass
class FollowUpEvent:
    content: str = ""
    type: str = "follow_up"


@dataclass
class ErrorEvent:
    content: str = ""
    type: str = "error"


AgentEvent = (
    StatusEvent
    | ResponseStartEvent
    | ResponseDelta
    | ResponseEndEvent
    | CardsEvent
    | FollowUpEvent
    | ErrorEvent
)
