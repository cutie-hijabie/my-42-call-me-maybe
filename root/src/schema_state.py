from enum import Enum
from typing import List, Dict
from .schema import InputDefinition, ParameterDefinition
from .json_state import JasonStateMacheine, JasonStates
from src import json_state


class SchemaStage(Enum):
    PROMPT = "prompt"
    NAME = "name"
    PARAMETERS = "parameters"


class SchemaState:
    def __init__(self, og_prompt: str):
        self.stage = SchemaStage.PROMPT
        self.function_name = None
        self.current_parameter: str | None = None
        self.partial_function_name = ""
        self.partial_parameter_name = ""
        self.completed_parameters = set()
        self.outer_key_candidates = ["prompt", "name", "parameters"]
        self.partial_key_candidate = ""
        self.current_outer_key = None
        self.og_prompt = og_prompt
        self.partial_og_prompt = ""

    def get_possible_key_candidates(self) -> List[str]:

        return self.outer_key_candidates

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

    def update(
        self,
        token_string: str,
        jason_state: JasonStateMacheine,
        function_names: List[str],
        parameters: Dict[str, ParameterDefinition],
    ) -> None:

        if (
            jason_state.state == JasonStates.IN_STRING
            and jason_state.string_role == "key"
        ) and (not len(jason_state.brackets) >= 2):
            self.partial_key_candidate += token_string

        if token_string == '"' and any(
            self.partial_key_candidate == key
            for key in self.outer_key_candidates
        ):
            self.outer_key_candidates.remove(self.partial_key_candidate)
            self.current_outer_key = self.partial_key_candidate
            self.partial_key_candidate = ""

        if self.stage == SchemaStage.PROMPT:
            if (
                jason_state.state == JasonStates.EXPECT_KEY
                and token_string == ","
            ):
                self.stage = SchemaStage.NAME
            elif self.current_outer_key == "prompt" and (
                (
                    jason_state.state == JasonStates.IN_STRING
                    and jason_state.string_role == "value"
                )
                or jason_state.state == JasonStates.IN_STRING_ESCAPE
            ):
                self.partial_og_prompt += token_string

        elif self.stage == SchemaStage.NAME:
            if (
                jason_state.state == JasonStates.IN_STRING
                and jason_state.string_role == "value"
            ):
                self.partial_function_name = (
                    self.partial_function_name + token_string.strip('"')
                )
                self.set_function_if_complete(function_names)
            elif (
                jason_state.state == JasonStates.EXPECT_KEY
                and token_string == ","
            ):
                self.stage = SchemaStage.PARAMETERS

        elif self.stage == SchemaStage.PARAMETERS:
            if len(jason_state.brackets) >= 2:
                if (
                    jason_state.state == JasonStates.IN_STRING
                    and jason_state.string_role == "key"
                ):
                    self.partial_parameter_name = (
                        self.partial_parameter_name + token_string.strip('"')
                    )
                    self.set_parameter_if_complete(parameters)
                elif (
                    jason_state.state == JasonStates.EXPECT_KEY
                    and token_string == ","
                ):
                    self.completed_parameters.add(self.current_parameter)
                    self.current_parameter = None
                    self.partial_parameter_name = ""
