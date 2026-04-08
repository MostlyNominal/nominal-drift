"""Base template class."""
from pydantic import BaseModel, Field
from datetime import datetime


class BaseTemplate(BaseModel, frozen=True):
    """Root template with type, timestamp, and version."""
    template_type: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    version: str = "1.0"

    model_config = {"frozen": True}
