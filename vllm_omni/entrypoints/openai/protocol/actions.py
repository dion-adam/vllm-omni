from typing import Any
from pydantic import BaseModel


class ActionGenerationRequest(BaseModel):
    task: str
    model: str | None = None
    state: list[float] | None = None
    images: dict[str, Any] | None = None
    extra_args: dict[str, Any] | None = None