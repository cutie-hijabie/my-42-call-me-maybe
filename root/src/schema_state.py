from enum import Enum
from typing import List, Dict
from .schema import InputDefinition, ParameterDefinition
from .json_state import JasonStateMacheine, JasonStates
import json


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
        self.required_outer_keys = frozenset(self.outer_key_candidates)
        self.completed_outer_fields: set[str] = set()
        self.partial_key_candidate = ""
        self.current_outer_key = None
        self.og_prompt = og_prompt
        self.json_og_prompt = json.dumps(og_prompt, ensure_ascii=False)
        self.partial_og_prompt = ""

    def get_possible_key_candidates(self) -> List[str]:

        return self.outer_key_candidates

    def get_possible_function_names(
        self, function_names: List[str]
    ) -> List[str]:
        return [
            name
            for name in function_names
            if json.dumps(name, ensure_ascii=False).startswith(
                self.partial_function_name
            )
        ]

    def set_function_if_complete(self, function_names: List[str]) -> None:
        for name in function_names:
            if self.partial_function_name == json.dumps(
                name, ensure_ascii=False
            ):
                self.function_name = name
                break

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
        return [
            name
            for name in parameters
            if name not in self.completed_parameters
            and json.dumps(name, ensure_ascii=False).startswith(
                self.partial_parameter_name
            )
        ]

    def set_parameter_if_complete(
        self, parameters: Dict[str, ParameterDefinition]
    ) -> None:
        for name in parameters:
            if (
                name not in self.completed_parameters
                and self.partial_parameter_name
                == json.dumps(name, ensure_ascii=False)
            ):
                self.current_parameter = name
                break

    def get_current_parameter_type(
        self, parameters: Dict[str, ParameterDefinition]
    ) -> str:
        if self.current_parameter:
            return parameters[self.current_parameter].type
        return ""

    def has_complete_outer_fields(self) -> bool:
        return self.completed_outer_fields == self.required_outer_keys

    def has_complete_parameters(
        self, parameters: Dict[str, ParameterDefinition]
    ) -> bool:
        return (
            self.function_name is not None
            and self.completed_parameters == set(parameters)
        )

    def update(
        self,
        token_string: str,
        jason_state: JasonStateMacheine,
        function_names: List[str],
        parameters: Dict[str, ParameterDefinition],
    ) -> None:

        if (
            jason_state.state == JasonStates.AFTER_VALUE
            and len(jason_state.brackets) == 1
        ):
            if (
                self.current_outer_key in ("prompt", "name")
                and token_string == '"'
            ) or (
                self.current_outer_key == "parameters" and token_string == "}"
            ):
                self.completed_outer_fields.add(self.current_outer_key)

        if (
            jason_state.state == JasonStates.IN_STRING
            and jason_state.string_role == "key"
            and token_string != '"'
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
            elif (
                self.current_outer_key == "prompt"
                and len(jason_state.brackets) == 1
                and jason_state.string_role == "value"
                and jason_state.state
                in (
                    JasonStates.IN_STRING,
                    JasonStates.IN_STRING_ESCAPE,
                    JasonStates.AFTER_VALUE,
                )
                and len(self.partial_og_prompt) < len(self.json_og_prompt)
            ):
                self.partial_og_prompt += token_string

        elif self.stage == SchemaStage.NAME:
            if (
                self.current_outer_key == "name"
                and len(jason_state.brackets) == 1
                and jason_state.string_role == "value"
                and (
                    jason_state.state
                    in (
                        JasonStates.IN_STRING,
                        JasonStates.IN_STRING_ESCAPE,
                    )
                    or (
                        token_string == '"'
                        and jason_state.state == JasonStates.AFTER_VALUE
                    )
                )
            ):
                self.partial_function_name += token_string

                if (
                    token_string == '"'
                    and jason_state.state == JasonStates.AFTER_VALUE
                ):
                    self.set_function_if_complete(function_names)

            elif (
                jason_state.state == JasonStates.EXPECT_KEY
                and token_string == ","
            ):
                self.stage = SchemaStage.PARAMETERS

        elif self.stage == SchemaStage.PARAMETERS:
            if (
                self.current_parameter is not None
                and jason_state.state == JasonStates.AFTER_VALUE
                and (
                    len(jason_state.brackets) == 2
                    or (token_string == "}" and len(jason_state.brackets) == 1)
                )
            ):
                self.completed_parameters.add(self.current_parameter)

            if len(jason_state.brackets) == 2:
                if jason_state.string_role == "key" and (
                    jason_state.state
                    in (
                        JasonStates.IN_STRING,
                        JasonStates.IN_STRING_ESCAPE,
                    )
                    or (
                        token_string == '"'
                        and jason_state.state == JasonStates.EXPECT_COLON
                    )
                ):
                    self.partial_parameter_name += token_string

                    if (
                        token_string == '"'
                        and jason_state.state == JasonStates.EXPECT_COLON
                    ):
                        self.set_parameter_if_complete(parameters)

                elif (
                    jason_state.state == JasonStates.EXPECT_KEY
                    and token_string == ","
                ):
                    if self.current_parameter is not None:
                        self.completed_parameters.add(self.current_parameter)

                    self.current_parameter = None
                    self.partial_parameter_name = ""
