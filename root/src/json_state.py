from enum import IntEnum


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
        self.brackets = []
        self.string_role = None
        self.digits_needed = 0

    def reset(self) -> None:
        self.state = JasonStates.START

    def transition(self, char: str) -> bool:

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
                return True
            elif char.isdigit():
                self.state = JasonStates.IN_NUMBER
                self.digit = True
                self.exponent = False
                self.decimal = False
                self.need_digit = False
                return True
            elif char == "-":
                self.state = JasonStates.IN_NUMBER
                self.digit = False
                self.exponent = False
                self.decimal = False
                self.need_digit = True
                return True

        elif self.state == JasonStates.EXPECT_KEY:
            if char == "}" and self.brackets[-1] == "{":
                self.brackets.pop()
                self.state = JasonStates.DONE
                return True
            elif char == '"':
                self.state = JasonStates.IN_STRING
                self.string_role = "key"
                return True

        elif self.state == JasonStates.IN_ARRAY:
            if char == "]" and self.brackets[-1] == "[":
                self.brackets.pop()
                self.state = JasonStates.DONE
                return True
            elif char.isdigit():
                self.state = JasonStates.IN_NUMBER
                self.digit = True
                self.exponent = False
                self.decimal = False
                self.need_digit = False
                return True
            elif char == "-":
                self.state = JasonStates.IN_NUMBER
                self.digit = False
                self.exponent = False
                self.decimal = False
                self.need_digit = True
                return True
            elif char == "t" or char == "n" or char == "f":
                self.state = JasonStates.IN_LITERAL
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
                return True

        elif self.state == JasonStates.IN_NUMBER:
            if char.isdigit():
                self.digit = True
                return True
            elif char == ".":
                if self.digit and not self.decimal and not self.exponent:
                    self.decimal = True
                    return True
            elif char in "eE":
                if self.digit and not self.exponent:
                    self.exponent = True
                    self.need_digit = True
                    return True
            elif char in "+-":
                if self.digit and self.exponent:
                    self.need_digit = True
                    return True
            elif char == "," and not self.need_digit:
                self.state = JasonStates.IN_ARRAY
                return True
            elif (
                char == "}"
                and not self.need_digit
                and self.brackets[-1] == "{"
            ):
                self.brackets.pop()
                self.state = JasonStates.DONE
                return True
            elif (
                char == "]"
                and not self.need_digit
                and self.brackets[-1] == "["
            ):
                self.brackets.pop()
                self.state = JasonStates.DONE
                return True

        elif self.state == JasonStates.IN_STRING:
            if not char == '"' and not char == "\\":
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
                return True
            elif char.isdigit():
                self.state = JasonStates.IN_NUMBER
                self.digit = True
                self.exponent = False
                self.decimal = False
                self.need_digit = False
                return True
            elif char == "-":
                self.state = JasonStates.IN_NUMBER
                self.digit = False
                self.exponent = False
                self.decimal = False
                self.need_digit = True
                return True

        return False
