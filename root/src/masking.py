from typing import List, Dict
from .json_state import JasonStateMacheine, JasonStates


def mask_invalid_tokens(
    logits: List[float],
    json_machine: JasonStateMacheine,
    vocab_map: Dict[int, str],
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

    return logits
