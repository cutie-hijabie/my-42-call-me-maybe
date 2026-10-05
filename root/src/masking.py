from typing import List, Dict
from .json_state import JasonStateMacheine, JasonStates
from .schema_state import SchemaState
from .schema import InputDefinition
from copy import copy as shallow_copy
import json


def mask_invalid_tokens(
    logits: List[float],
    json_machine: JasonStateMacheine,
    vocab_map: Dict[int, str],
    schema_state: SchemaState,
    function_names: List[str],
    definitions: List[InputDefinition],
) -> List[float]:

    serialized_function_names = [
        json.dumps(name, ensure_ascii=False) for name in function_names
    ]

    for token_id in range(len(logits)):
        if token_id not in vocab_map:
            logits[token_id] = float("-inf")
            continue
        token_string = vocab_map[token_id]
        if not token_string:
            logits[token_id] = float("-inf")
            continue
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
            and json_machine.digits_needed == 0
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

        if (
            json_machine.state == JasonStates.IN_STRING
            and json_machine.string_role == "value"
            and token_string.isprintable()
            and '"' not in token_string
            and "\\" not in token_string
        ):
            depth = len(json_machine.brackets)

            if depth == 1 and schema_state.current_outer_key == "prompt":
                if not schema_state.json_og_prompt.startswith(
                    schema_state.partial_og_prompt + token_string
                ):
                    logits[token_id] = float("-inf")
                continue

            if depth == 1 and schema_state.current_outer_key == "name":
                if not any(
                    name.startswith(
                        schema_state.partial_function_name + token_string
                    )
                    for name in serialized_function_names
                ):
                    logits[token_id] = float("-inf")
                continue

            if depth == 2:
                parameters = schema_state.get_parameters(definitions)
                if (
                    schema_state.get_current_parameter_type(parameters)
                    == "string"
                ):
                    continue

        candidate_machine = shallow_copy(json_machine)
        candidate_machine.brackets = json_machine.brackets.copy()

        candidate_schema = shallow_copy(schema_state)
        candidate_schema.outer_key_candidates = (
            schema_state.outer_key_candidates.copy()
        )
        candidate_schema.completed_parameters = (
            schema_state.completed_parameters.copy()
        )
        candidate_schema.completed_outer_fields = (
            schema_state.completed_outer_fields.copy()
        )

        candidate_valid = True

        for char in token_string:
            state = candidate_machine.state
            if char in " \t\r\n" and state not in (
                JasonStates.IN_STRING,
                JasonStates.IN_STRING_ESCAPE,
            ):
                candidate_valid = False
                break
            depth = len(candidate_machine.brackets)
            candidate_parameters = candidate_schema.get_parameters(definitions)

            if (
                state == JasonStates.START
                and char not in " \t\r\n"
                and char != "{"
            ):
                candidate_valid = False
                break

            if depth == 1 and state == JasonStates.EXPECT_KEY:
                if (
                    char == '"'
                    and candidate_schema.stage.value
                    not in candidate_schema.outer_key_candidates
                ):
                    candidate_valid = False
                    break

            if (
                depth == 1
                and state == JasonStates.IN_STRING
                and candidate_machine.string_role == "key"
            ):
                expected_key = candidate_schema.stage.value
                partial_key = candidate_schema.partial_key_candidate

                key_valid = (
                    partial_key == expected_key
                    if char == '"'
                    else expected_key.startswith(partial_key + char)
                )

                if (
                    expected_key not in candidate_schema.outer_key_candidates
                    or not key_valid
                ):
                    candidate_valid = False
                    break
            if (
                depth == 1
                and state == JasonStates.EXPECT_VALUE
                and char not in " \t\r\n"
            ):
                expected_start = (
                    "{"
                    if candidate_schema.current_outer_key == "parameters"
                    else '"'
                )

                if char != expected_start:
                    candidate_valid = False
                    break
            if (
                depth == 1
                and candidate_schema.current_outer_key == "prompt"
                and (
                    state == JasonStates.EXPECT_VALUE
                    or (
                        state
                        in (
                            JasonStates.IN_STRING,
                            JasonStates.IN_STRING_ESCAPE,
                        )
                        and candidate_machine.string_role == "value"
                    )
                )
                and not (
                    state == JasonStates.EXPECT_VALUE and char in " \t\r\n"
                )
            ):
                candidate_prompt = candidate_schema.partial_og_prompt + char

                if not candidate_schema.json_og_prompt.startswith(
                    candidate_prompt
                ):
                    candidate_valid = False
                    break
            if (
                depth == 1
                and candidate_schema.current_outer_key == "name"
                and (
                    state == JasonStates.EXPECT_VALUE
                    or (
                        state
                        in (
                            JasonStates.IN_STRING,
                            JasonStates.IN_STRING_ESCAPE,
                        )
                        and candidate_machine.string_role == "value"
                    )
                )
                and not (
                    state == JasonStates.EXPECT_VALUE and char in " \t\r\n"
                )
            ):
                candidate_name = candidate_schema.partial_function_name + char

                if not any(
                    name.startswith(candidate_name)
                    for name in serialized_function_names
                ):
                    candidate_valid = False
                    break
            if depth == 2 and (
                (
                    state == JasonStates.EXPECT_KEY
                    and char not in " \t\r\n"
                    and char != "}"
                )
                or (
                    state
                    in (
                        JasonStates.IN_STRING,
                        JasonStates.IN_STRING_ESCAPE,
                    )
                    and candidate_machine.string_role == "key"
                )
            ):
                candidate_key = candidate_schema.partial_parameter_name + char

                if not any(
                    name not in candidate_schema.completed_parameters
                    and json.dumps(name, ensure_ascii=False).startswith(
                        candidate_key
                    )
                    for name in candidate_parameters
                ):
                    candidate_valid = False
                    break
            if depth == 2 and state in (
                JasonStates.EXPECT_VALUE,
                JasonStates.IN_NUMBER,
            ):
                candidate_type = candidate_schema.get_current_parameter_type(
                    candidate_parameters
                )

                if state == JasonStates.EXPECT_VALUE and char not in " \t\r\n":
                    valid_start = (
                        (candidate_type == "string" and char == '"')
                        or (
                            candidate_type in ("number", "integer")
                            and char in "-0123456789"
                        )
                        or (candidate_type == "boolean" and char in "tf")
                        or (candidate_type == "null" and char == "n")
                    )

                    if not valid_start:
                        candidate_valid = False
                        break

                elif (
                    candidate_type == "integer"
                    and state == JasonStates.IN_NUMBER
                    and char in ".eE"
                ):
                    candidate_valid = False
                    break
            if (
                char == "}"
                and len(candidate_machine.brackets) == 1
                and candidate_machine.state
                in (JasonStates.EXPECT_KEY, JasonStates.AFTER_VALUE)
                and not candidate_schema.has_complete_outer_fields()
            ):
                candidate_valid = False
                break

            if not candidate_machine.transition(char):
                candidate_valid = False
                break

            candidate_schema.update(
                char,
                candidate_machine,
                function_names,
                candidate_parameters,
            )

            if (
                char == "}"
                and depth == 2
                and len(candidate_machine.brackets) == 1
                and not candidate_schema.has_complete_parameters(
                    candidate_parameters
                )
            ):
                candidate_valid = False
                break

            if (
                char == ","
                and candidate_machine.state == JasonStates.EXPECT_KEY
                and (
                    (
                        depth == 2
                        and candidate_schema.has_complete_parameters(
                            candidate_parameters
                        )
                    )
                    or (
                        depth == 1
                        and candidate_schema.has_complete_outer_fields()
                    )
                )
            ):
                candidate_valid = False
                break

        if not candidate_valid:
            logits[token_id] = float("-inf")
            continue

    return logits
