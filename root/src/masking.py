from typing import List, Dict
from .json_state import JasonStateMacheine, JasonStates
from .schema_state import SchemaState, SchemaStage
from .schema import InputDefinition


def mask_invalid_tokens(
    logits: List[float],
    json_machine: JasonStateMacheine,
    vocab_map: Dict[int, str],
    schema_state: SchemaState,
    function_names: List[str],
    definitions: List[InputDefinition],
) -> List[float]:

    for token_id in range(len(logits)):
        if token_id not in vocab_map:
            logits[token_id] = float("-inf")
            continue
        token_string = vocab_map[token_id]
        if (
            json_machine.state == JasonStates.EXPECT_COLON
            and not token_string.startswith(
                (
                    ":",
                    " ",
                    "\t",
                    "\n",
                    "\r",
                )
            )
        ):
            logits[token_id] = float("-inf")
            continue
        elif (
            json_machine.state == JasonStates.IN_STRING_ESCAPE
            and not token_string.startswith(
                ("\\", "u", "n", "r", "t", "b", "f", '"', "/")
            )
        ):
            logits[token_id] = float("-inf")
            continue
        elif (
            json_machine.state == JasonStates.AFTER_VALUE
            and not token_string.startswith(
                (
                    " ",
                    "}",
                    "]",
                    ",",
                    "\t",
                    "\n",
                    "\r",
                )
            )
        ):
            logits[token_id] = float("-inf")
            continue
        elif (
            json_machine.state == JasonStates.EXPECT_KEY
            and not token_string.startswith(
                (
                    "}",
                    '"',
                    " ",
                    "\t",
                    "\n",
                    "\r",
                )
            )
        ):
            logits[token_id] = float("-inf")
            continue
        elif (
            json_machine.state == JasonStates.DONE
            and not token_string.startswith((" ", "\t", "\n", "\r"))
        ):
            logits[token_id] = float("-inf")
            continue
        elif (
            json_machine.state == JasonStates.START
            or json_machine.state == JasonStates.EXPECT_VALUE
        ) and not token_string.startswith(
            (
                "[",
                "{",
                "1",
                '"',
                "t",
                "n",
                "f",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "0",
                "\t",
                "\n",
                "\r",
                "-",
                " ",
            )
        ):
            logits[token_id] = float("-inf")
            continue
        elif (
            json_machine.state == JasonStates.IN_ARRAY
            and not token_string.startswith(
                (
                    "[",
                    "{",
                    "1",
                    '"',
                    "t",
                    "n",
                    "f",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "0",
                    "]",
                    "\t",
                    "\n",
                    "\r",
                    "-",
                    " ",
                )
            )
        ):
            logits[token_id] = float("-inf")
            continue

        if not json_machine.validate_candidate(token_string):
            logits[token_id] = float("-inf")
            continue

        if schema_state.stage == SchemaStage.NAME and (
            json_machine.state == JasonStates.EXPECT_VALUE
            or (
                json_machine.state == JasonStates.IN_STRING
                and json_machine.string_role == "value"
            )
        ):
            copy = schema_state.partial_function_name + token_string.strip('"')
            is_valid = any(
                function.startswith(copy) for function in function_names
            )
            if '"' in token_string:
                if (
                    json_machine.state == JasonStates.IN_STRING
                    and json_machine.string_role == "value"
                ):
                    is_valid = any(
                        copy == function for function in function_names
                    )
                elif (
                    json_machine.state == JasonStates.EXPECT_VALUE
                    and token_string.count('"') >= 2
                ):
                    is_valid = any(
                        copy == function for function in function_names
                    )

            if not is_valid:
                logits[token_id] = float("-inf")
                continue

        if (
            schema_state.stage == SchemaStage.PARAMETERS
            and json_machine.state == JasonStates.EXPECT_VALUE
        ):
            parameters = schema_state.get_parameters(definitions)
            parameter_type = schema_state.get_current_parameter_type(
                parameters
            )
            copy = token_string.lstrip()
            if parameter_type == "string" and not copy.startswith('"'):
                logits[token_id] = float("-inf")
                continue
            elif (
                copy
                and parameter_type == "number"
                and not (copy.startswith("-") or copy[0] in "1234567890")
            ):
                logits[token_id] = float("-inf")
                continue

        elif schema_state.stage == SchemaStage.PARAMETERS and (
            (
                json_machine.state == JasonStates.IN_STRING
                and json_machine.string_role == "key"
            )
            or (json_machine.state == JasonStates.EXPECT_KEY)
        ):
            parameters = schema_state.get_parameters(definitions)
            copy = schema_state.partial_parameter_name + token_string.strip(
                '"'
            )
            is_valid = any(
                parameter.startswith(copy)
                and parameter not in schema_state.completed_parameters
                for parameter in parameters
            )
            if '"' in token_string:
                if (
                    json_machine.state == JasonStates.IN_STRING
                    and json_machine.string_role == "key"
                ):
                    is_valid = any(
                        copy == parameter
                        and parameter not in schema_state.completed_parameters
                        for parameter in parameters
                    )
                elif (
                    json_machine.state == JasonStates.EXPECT_KEY
                    and token_string.count('"') >= 2
                ):
                    is_valid = any(
                        copy == parameter
                        and parameter not in schema_state.completed_parameters
                        for parameter in parameters
                    )
            if not is_valid:
                logits[token_id] = float("-inf")
                continue

    return logits
