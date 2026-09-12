from typing import Dict, List
from pydantic import BaseModel, RootModel


class ParameterDefinition(BaseModel):
    type: str


class ReturnsDefinition(BaseModel):
    type: str


class InputDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnsDefinition


class InputFileDefinition(RootModel[List[InputDefinition]]):
    pass
