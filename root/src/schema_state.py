from enum import Enum
from typing import List, Dict
from .schema import InputDefinition, ParameterDefinition


class SchemaStage(Enum):
    PROMPT = "prompt"
    NAME = "name"
    PARAMETERS = "parameters"


class SchemaState:
    def __init__(self):
        self.stage = SchemaStage.PROMPT
        self.function_name = None
        self.current_parameter: str | None = None
        self.partial_function_name = ""
        self.partial_parameter_name = ""

    def get_possible_function_names(
        self, function_names: List[str]
    ) -> List[str]:
        result = []
        for name in function_names:
            if name.startswith(self.partial_function_name):
                result.append(name)

        return result

    def set_function_if_complete(self, function_names: List[str]) -> None:
        if self.partial_function_name in function_names:
            self.function_name = self.partial_function_name

    def get_parameters(
        self, definitions: List[InputDefinition]
    ) -> Dict[str, ParameterDefinition]:

        result = {}

        for definition in definitions:
            if self.function_name == definition.name:
                return definition.parameters

        return result

    def get_possible_parameter_names(
        self, parameters: Dict[str, ParameterDefinition]
    ) -> List[str]:
        result = []
        for parameter in parameters:
            if parameter.startswith(self.partial_parameter_name):
                result.append(parameter)

        return result

    def set_parameter_if_complete(
        self, parameters: Dict[str, ParameterDefinition]
    ) -> None:

        if self.partial_parameter_name in parameters:
            self.current_parameter = self.partial_parameter_name

    def get_current_parameter_type(
        self, parameters: Dict[str, ParameterDefinition]
    ) -> str:
        if self.current_parameter:
            return parameters[self.current_parameter].type
        return ""
