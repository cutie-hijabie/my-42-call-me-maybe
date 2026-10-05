from typing import Dict, List, Literal
from pydantic import BaseModel, RootModel


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
