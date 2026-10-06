from enum import IntEnum
import copy
from typing import Tuple
from typing_extensions import List


class JasonStates(IntEnum):
    START = 0
    EXPECT_KEY = 1
    EXPECT_COLON = 2
    EXPECT_VALUE = 3
    IN_STRING = 4
    IN_STRING_ESCAPE = 5
    IN_NUMBER = 6
    AFTER_VALUE = 7
    IN_ARRAY = 8
    DONE = 9
    IN_LITERAL = 10


class JasonStateMacheine:
    def __init__(self) -> None:
        self.state = JasonStates.START
        self.brackets: List[str] = []
        self.string_role: str | None = None
        self.digits_needed = 0
        self.literal_target: str | None = None
        self.literal_pos = 0
        self.is_leading_zero = False
        self.exponent = False
        self.has_exponent_sign = False
        self.digit = True
        self.decimal = False
        self.need_digit = False
        self.allow_empty_object = False

    def reset(self) -> None:
        self.state = JasonStates.START
        self.brackets = []
        self.string_role = None
        self.digits_needed = 0
        self.literal_target = None
        self.literal_pos = 0
        self.is_leading_zero = False
        self.exponent = False
        self.has_exponent_sign = False
        self.digit = True
        self.decimal = False
        self.need_digit = False
        self.allow_empty_object = False

    def transition(self, char: str) -> bool:

        if char in "\t\n\r " and not (
            self.state == JasonStates.IN_STRING
            or self.state == JasonStates.IN_STRING_ESCAPE
        ):
            if self.state == JasonStates.IN_LITERAL:
                return False

            if self.state == JasonStates.IN_NUMBER:
                if self.need_digit:
                    return False

                self.state = (
                    JasonStates.AFTER_VALUE
                    if self.brackets
                    else JasonStates.DONE
                )
            return True

        if self.digits_needed > 0:
            if char in "0123456789abcdefABCDEF":
                self.digits_needed -= 1
                if self.digits_needed == 0:
                    self.state = JasonStates.IN_STRING
                return True
            else:
                return False

        if self.state == JasonStates.START:
            if char == "{":
                self.state = JasonStates.EXPECT_KEY
                self.brackets.append(char)
                self.allow_empty_object = True
                return True
            elif char == "[":
                self.state = JasonStates.IN_ARRAY
                self.brackets.append(char)
                return True
            elif char == '"':
                self.state = JasonStates.IN_STRING
                return True
            elif char == "t" or char == "n" or char == "f":
                self.state = JasonStates.IN_LITERAL
                self.literal_pos = 1
                if char == "t":
                    self.literal_target = "true"
                elif char == "n":
                    self.literal_target = "null"
                else:
                    self.literal_target = "false"
                return True
            elif char in "0123456789":
                self.state = JasonStates.IN_NUMBER
                self.digit = True
                self.exponent = False
                self.decimal = False
                self.need_digit = False
                self.is_leading_zero = False
                self.has_exponent_sign = False
                if int(char) == 0:
                    self.is_leading_zero = True
                return True
            elif char == "-":
                self.state = JasonStates.IN_NUMBER
                self.digit = False
                self.exponent = False
                self.decimal = False
                self.need_digit = True
                self.has_exponent_sign = False
                self.is_leading_zero = False
                return True

        elif self.state == JasonStates.EXPECT_KEY:
            if (
                char == "}"
                and self.brackets
                and self.brackets[-1] == "{"
                and self.allow_empty_object
            ):
                self.brackets.pop()
                if not self.brackets:
                    self.state = JasonStates.DONE
                else:
                    self.state = JasonStates.AFTER_VALUE
                return True
            elif char == '"':
                self.state = JasonStates.IN_STRING
                self.string_role = "key"
                return True

        elif self.state == JasonStates.IN_ARRAY:
            if char == "]" and self.brackets and self.brackets[-1] == "[":
                self.brackets.pop()
                if not self.brackets:
                    self.state = JasonStates.DONE
                else:
                    self.state = JasonStates.AFTER_VALUE
                return True
            elif char in "0123456789":
                self.state = JasonStates.IN_NUMBER
                self.digit = True
                self.exponent = False
                self.decimal = False
                self.need_digit = False
                self.is_leading_zero = False
                self.has_exponent_sign = False
                if int(char) == 0:
                    self.is_leading_zero = True
                return True
            elif char == "-":
                self.state = JasonStates.IN_NUMBER
                self.digit = False
                self.exponent = False
                self.decimal = False
                self.need_digit = True
                self.is_leading_zero = False
                self.has_exponent_sign = False
                return True
            elif char == "t" or char == "n" or char == "f":
                self.state = JasonStates.IN_LITERAL
                self.literal_pos = 1
                if char == "t":
                    self.literal_target = "true"
                elif char == "n":
                    self.literal_target = "null"
                else:
                    self.literal_target = "false"
                return True
            elif char == "[":
                self.state = JasonStates.IN_ARRAY
                self.brackets.append(char)
                return True
            elif char == '"':
                self.state = JasonStates.IN_STRING
                self.string_role = "value"
                return True
            elif char == "{":
                self.state = JasonStates.EXPECT_KEY
                self.brackets.append(char)
                self.allow_empty_object = True
                return True

        elif self.state == JasonStates.IN_NUMBER:
            if char in "0123456789":
                if self.is_leading_zero:
                    return False
                if int(char) == 0 and not self.digit:
                    self.is_leading_zero = True
                self.digit = True
                if self.need_digit:
                    self.need_digit = False
                return True
            elif char == ".":
                if self.digit and not self.decimal and not self.exponent:
                    self.need_digit = True
                    self.decimal = True
                    if self.is_leading_zero:
                        self.is_leading_zero = False
                    return True
            elif char in "eE" and not self.need_digit:
                if self.digit and not self.exponent:
                    self.exponent = True
                    self.need_digit = True
                    if self.is_leading_zero:
                        self.is_leading_zero = False
                    return True
            elif char in "+-":
                if (
                    self.digit
                    and self.exponent
                    and not self.has_exponent_sign
                    and self.need_digit
                ):
                    self.need_digit = True
                    self.has_exponent_sign = True
                    return True
            elif char == "," and not self.need_digit:
                self.state = JasonStates.AFTER_VALUE
                return self.transition(char)
            elif (
                char == "}"
                and self.brackets
                and not self.need_digit
                and self.brackets[-1] == "{"
            ):
                self.state = JasonStates.AFTER_VALUE
                return self.transition(char)
            elif (
                char == "]"
                and self.brackets
                and not self.need_digit
                and self.brackets[-1] == "["
            ):
                self.state = JasonStates.AFTER_VALUE
                return self.transition(char)

        elif self.state == JasonStates.IN_STRING:
            if ord(char) < 32:
                return False
            elif not char == '"' and not char == "\\":
                return True
            elif char == '"':
                if self.string_role == "value":
                    self.state = JasonStates.AFTER_VALUE
                    return True
                elif self.string_role == "key":
                    self.state = JasonStates.EXPECT_COLON
                    return True
                else:
                    self.state = JasonStates.DONE
                    return True
            elif char == "\\":
                self.state = JasonStates.IN_STRING_ESCAPE
                return True

        elif self.state == JasonStates.IN_STRING_ESCAPE:
            if char in '"\\/bfnrt':
                self.state = JasonStates.IN_STRING
                return True
            elif char == "u":
                self.digits_needed = 4
                return True

        elif self.state == JasonStates.EXPECT_COLON:
            if char == ":":
                self.state = JasonStates.EXPECT_VALUE
                return True

        elif self.state == JasonStates.EXPECT_VALUE:
            if char == "{":
                self.state = JasonStates.EXPECT_KEY
                self.brackets.append(char)
                self.allow_empty_object = True
                return True
            elif char == "[":
                self.state = JasonStates.IN_ARRAY
                self.brackets.append(char)
                return True
            elif char == '"':
                self.string_role = "value"
                self.state = JasonStates.IN_STRING
                return True
            elif char == "t" or char == "n" or char == "f":
                self.state = JasonStates.IN_LITERAL
                self.literal_pos = 1
                if char == "t":
                    self.literal_target = "true"
                elif char == "n":
                    self.literal_target = "null"
                else:
                    self.literal_target = "false"
                return True
            elif char in "0123456789":
                self.state = JasonStates.IN_NUMBER
                self.digit = True
                self.exponent = False
                self.decimal = False
                self.need_digit = False
                self.is_leading_zero = False
                self.has_exponent_sign = False
                if int(char) == 0:
                    self.is_leading_zero = True
                return True
            elif char == "-":
                self.state = JasonStates.IN_NUMBER
                self.digit = False
                self.exponent = False
                self.decimal = False
                self.need_digit = True
                self.has_exponent_sign = False
                self.is_leading_zero = False
                return True

        elif self.state == JasonStates.AFTER_VALUE:
            if char == ",":
                if self.brackets and self.brackets[-1] == "{":
                    self.state = JasonStates.EXPECT_KEY
                    self.allow_empty_object = False
                    return True
                elif self.brackets and self.brackets[-1] == "[":
                    self.state = JasonStates.EXPECT_VALUE
                    return True
            elif char == "]" and self.brackets and self.brackets[-1] == "[":
                self.brackets.pop()
                if not self.brackets:
                    self.state = JasonStates.DONE
                else:
                    self.state = JasonStates.AFTER_VALUE
                return True
            elif char == "}" and self.brackets and self.brackets[-1] == "{":
                self.brackets.pop()
                if not self.brackets:
                    self.state = JasonStates.DONE
                else:
                    self.state = JasonStates.AFTER_VALUE
                return True

        elif self.state == JasonStates.IN_LITERAL:
            if (
                self.literal_target
                and char == self.literal_target[self.literal_pos]
            ):
                self.literal_pos += 1
                if (
                    self.literal_target
                    and len(self.literal_target) == self.literal_pos
                ):
                    self.state = JasonStates.AFTER_VALUE
                return True

        return False

    def is_complete(self) -> bool:

        if self.state == JasonStates.START:
            return False
        elif self.state == JasonStates.EXPECT_KEY:
            return False
        elif self.state == JasonStates.IN_ARRAY:
            return False
        elif self.state == JasonStates.IN_STRING_ESCAPE:
            return False
        elif self.state == JasonStates.IN_STRING:
            return False
        elif self.need_digit:
            return False
        elif self.state == JasonStates.EXPECT_COLON:
            return False
        elif self.state == JasonStates.EXPECT_VALUE:
            return False
        elif self.state == JasonStates.AFTER_VALUE and self.brackets:
            return False
        elif self.state == JasonStates.IN_LITERAL:
            return False
        elif self.state == JasonStates.IN_NUMBER and self.brackets:
            return False
        return True

    def validate_candidate(self, candidate: str) -> bool:
        machine_copy = copy.copy(self)
        machine_copy.brackets = self.brackets.copy()

        for char in candidate:
            if not machine_copy.transition(char):
                return False

        return True

    def trace(
        self, input: str
    ) -> List[Tuple[str, "JasonStates", bool, List[str]]]:
        final = []

        machine_copy = copy.deepcopy(self)

        for char in input:
            result = machine_copy.transition(char)
            legal_chars = machine_copy.legal_next_chars()
            final.append((char, machine_copy.state, result, legal_chars))

        return final

    def legal_next_chars(self) -> List[str]:
        candidates = '{}[]":,\\-+.0123456789tfnEe \t\n\rabcdefABCDEFubfnrt/'
        valid = []

        for char in candidates:
            result = self.validate_candidate(char)
            if result:
                valid.append(char)

        return valid
