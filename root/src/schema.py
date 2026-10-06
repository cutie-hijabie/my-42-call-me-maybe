from typing import Dict, List, Literal
from pydantic import BaseModel, RootModel, field_validator


class ParameterDefinition(BaseModel):
    type: Literal["string", "number", "integer", "boolean", "null"]


class ReturnsDefinition(BaseModel):
    type: str


class InputDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnsDefinition


class InputFileDefinition(RootModel[List[InputDefinition]]):
    pass


class InputPrompt(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject empty prompts while preserving the original text."""
        if not value.strip():
            raise ValueError("Prompt must not be empty or whitespace-only.")
        return value


class InputFilePrompts(RootModel[List[InputPrompt]]):
    pass
