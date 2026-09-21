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

        if schema_state.stage == SchemaStage.NAME:
            copy = (
                schema_state.partial_function_name
                + token_string.split('"', 1)[0]
            )
            if '"' in token_string:
                is_valid = True if copy in function_names else False
            else:
                is_valid = any(
                    function.startswith(copy) for function in function_names
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
            elif parameter_type == "number" and not (
                copy.startswith("-") or copy.isdigit()
            ):
                logits[token_id] = float("-inf")
                continue

        elif schema_state.stage == SchemaStage.PARAMETERS:
            parameters = schema_state.get_parameters(definitions)
            copy = (
                schema_state.partial_parameter_name
                + token_string.split('"', 1)[0]
            )
            if '"' in token_string:
                is_valid = True if copy in parameters else False
            else:
                is_valid = any(
                    parameter.startswith(copy) for parameter in parameters
                )
            if not is_valid:
                logits[token_id] = float("-inf")
                continue

    return logits
