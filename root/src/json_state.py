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


class JasonStateMacheine:
    def __init__(self) -> None:
        self.state = JasonStates.START

    def reset(self) -> None:
        self.state = JasonStates.START

    def transition(self, char: str) -> bool:

        return True
