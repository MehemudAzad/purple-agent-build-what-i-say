from typing import Any
from pydantic import BaseModel, HttpUrl

class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl] # role-endpoint mapping
    config: dict[str, Any]

class EvalResult(BaseModel):
    accuracy: float
    avg_questions_per_instruction: float
    overall_avg_score: float