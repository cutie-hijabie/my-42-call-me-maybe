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

    parameters = schema_state.get_parameters(definitions)
    parameter_type = schema_state.get_current_parameter_type(parameters)

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
            json_machine.state == JasonStates.EXPECT_KEY
            or (
                json_machine.state == JasonStates.IN_STRING
                and json_machine.string_role == "key"
            )
        ) and not len(json_machine.brackets) >= 2:
            copy = schema_state.partial_key_candidate + token_string.strip('"')
            is_valid = any(
                candidate.startswith(copy)
                for candidate in schema_state.outer_key_candidates
            )
            if '"' in token_string:
                if json_machine.state == JasonStates.IN_STRING or (
                    json_machine.state == JasonStates.EXPECT_KEY
                    and token_string.count('"') >= 2
                ):
                    is_valid = any(
                        copy == candidate
                        for candidate in schema_state.outer_key_candidates
                    )

            if not is_valid:
                logits[token_id] = float("-inf")
                continue

        if schema_state.stage == SchemaStage.PROMPT:
            if (
                json_machine.state == JasonStates.START
                and not token_string.startswith("{")
            ):
                logits[token_id] = float("-inf")
                continue
            if (
                json_machine.state == JasonStates.EXPECT_KEY
                or (
                    json_machine.state == JasonStates.IN_STRING
                    and json_machine.string_role == "key"
                )
            ) and not len(json_machine.brackets) >= 2:
                copy = schema_state.partial_key_candidate + token_string.strip(
                    '"'
                )
                is_valid = True if "prompt".startswith(copy) else False
                if '"' in token_string:
                    if json_machine.state == JasonStates.IN_STRING or (
                        json_machine.state == JasonStates.EXPECT_KEY
                        and token_string.count('"') >= 2
                    ):
                        is_valid = copy == "prompt"
                if not is_valid:
                    logits[token_id] = float("-inf")
                    continue

            if (schema_state.current_outer_key == "prompt") and (
                (
                    json_machine.state == JasonStates.IN_STRING
                    and json_machine.string_role == "value"
                )
                or (json_machine.state == JasonStates.EXPECT_VALUE)
            ):
                copy = schema_state.partial_og_prompt + token_string.strip('"')
                is_valid = schema_state.og_prompt.startswith(copy)
                if '"' in token_string:
                    if (
                        json_machine.state == JasonStates.IN_STRING
                        and json_machine.string_role == "value"
                    ) or (
                        json_machine.state == JasonStates.EXPECT_VALUE
                        and token_string.count('"') >= 2
                    ):
                        is_valid = copy == schema_state.og_prompt

                if not is_valid:
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

        elif (
            schema_state.stage == SchemaStage.PARAMETERS
            and len(json_machine.brackets) >= 2
            and (
                (
                    json_machine.state == JasonStates.IN_STRING
                    and json_machine.string_role == "key"
                )
                or (json_machine.state == JasonStates.EXPECT_KEY)
            )
        ):
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

        if not json_machine.validate_candidate(token_string):
            logits[token_id] = float("-inf")
            continue
    return logits
