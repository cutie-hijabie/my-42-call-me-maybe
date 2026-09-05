import pytest

from src.json_state import JasonStateMacheine


def assert_valid_json(text):
    machine = JasonStateMacheine()

    for index, char in enumerate(text):
        assert machine.transition(char) is True, (
            f"Rejected valid JSON at index {index}: {char!r}\n"
            f"Input: {text!r}\n"
            f"State: {machine.state}"
        )

    assert machine.is_complete() is True, (
        f"Accepted all characters but JSON was considered incomplete.\n"
        f"Input: {text!r}\n"
        f"Final state: {machine.state}"
    )


def assert_invalid_json(text):
    machine = JasonStateMacheine()

    for char in text:
        if machine.transition(char) is False:
            return

    assert machine.is_complete() is False, (
        f"Invalid JSON was considered complete: {text!r}\n"
        f"Final state: {machine.state}"
    )


def assert_invalid_at(text, bad_index):
    machine = JasonStateMacheine()

    for index, char in enumerate(text):
        result = machine.transition(char)

        if index < bad_index:
            assert result is True, (
                f"Rejected too early at index {index}: {char!r}\n"
                f"Input: {text!r}"
            )

        elif index == bad_index:
            assert result is False, (
                f"Expected rejection at index {bad_index}: {char!r}\n"
                f"Input: {text!r}"
            )
            return

    pytest.fail(
        f"Input never reached expected invalid index {bad_index}: {text!r}"
    )


# =========================================================
# NUMBERS
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-0",
        "1",
        "-1",
        "12",
        "-12",
        "123456789",
        "0.5",
        "-0.5",
        "12.34",
        "-12.34",
        "1e5",
        "1E5",
        "1e-5",
        "1e+5",
        "1E-5",
        "1E+5",
        "1.5e2",
        "1.5e-2",
        "-1.5e+2",
        "0e0",
        "-0e0",
    ],
)
def test_valid_numbers(value):
    assert_valid_json(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "-",
        "01",
        "-01",
        "00",
        "-00",
        "1.",
        "-1.",
        ".5",
        "-.5",
        "1.e5",
        "1e",
        "1E",
        "1e-",
        "1e+",
        "1e--5",
        "1e++5",
        "1e+-5",
        "1e-+5",
        "1e5-6",
        "1e5+6",
        "1..2",
        "1.2.3",
        "1e2e3",
    ],
)
def test_invalid_numbers(value):
    assert_invalid_json(value)


# =========================================================
# STRINGS
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        '""',
        '"hello"',
        '"hello world"',
        '"123"',
        '"true"',
        '"{}[]"',
        '"hello \\"world\\""',
        '"backslash: \\\\"',
        '"slash: \\/"',
        '"newline: \\n"',
        '"tab: \\t"',
        '"return: \\r"',
        '"backspace: \\b"',
        '"formfeed: \\f"',
        '"unicode: \\u0041"',
        '"unicode: \\uabcd"',
        '"unicode: \\uABCD"',
    ],
)
def test_valid_strings(value):
    assert_valid_json(value)


@pytest.mark.parametrize(
    "value",
    [
        '"',
        '"hello',
        '"hello\\',
        '"\\x"',
        '"\\q"',
        '"\\u"',
        '"\\u0"',
        '"\\u00"',
        '"\\u000"',
        '"\\u000G"',
    ],
)
def test_invalid_or_incomplete_strings(value):
    assert_invalid_json(value)


def test_string_with_unescaped_newline_is_invalid():
    assert_invalid_json('"hello\nworld"')


# =========================================================
# LITERALS
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        "true",
        "false",
        "null",
    ],
)
def test_valid_literals(value):
    assert_valid_json(value)


@pytest.mark.parametrize(
    "value",
    [
        "t",
        "tr",
        "tru",
        "f",
        "fa",
        "fal",
        "fals",
        "n",
        "nu",
        "nul",
        "True",
        "False",
        "NULL",
        "truth",
        "none",
    ],
)
def test_invalid_or_incomplete_literals(value):
    assert_invalid_json(value)


# =========================================================
# ARRAYS
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        "[]",
        "[1]",
        "[1,2]",
        "[1,2,3]",
        '["a"]',
        '["a","b"]',
        "[true]",
        "[false]",
        "[null]",
        "[true,false,null]",
        '[1,true,null,"hello"]',
        "[[]]",
        "[[1]]",
        "[[1,2],[3,4]]",
        "[{},{}]",
        '[{"a":1}]',
        "[1,[2,[3]]]",
    ],
)
def test_valid_arrays(value):
    assert_valid_json(value)


@pytest.mark.parametrize(
    "value",
    [
        "[",
        "[1",
        "[1,",
        "[1,2",
        "[,]",
        "[,1]",
        "[1,,2]",
        "[1,]",
        "[1}",
        "[true,]",
        "[null,]",
        '["a",]',
        "[[1]",
    ],
)
def test_invalid_or_incomplete_arrays(value):
    assert_invalid_json(value)


# =========================================================
# OBJECTS
# =========================================================


def test_json_with_whitespace():
    assert_valid_json('{ "a" : 1 }')


@pytest.mark.parametrize(
    "value",
    [
        "{}",
        '{"a":1}',
        '{"a":"hello"}',
        '{"a":true}',
        '{"a":false}',
        '{"a":null}',
        '{"a":[]}',
        '{"a":{}}',
        '{"a":1,"b":2}',
        '{"a":1,"b":"two","c":true}',
        '{"nested":{"value":1}}',
        '{"array":[1,2,3]}',
        '{"a":{"b":{"c":1}}}',
        '{"a":[{"b":1},{"c":2}]}',
    ],
)
def test_valid_objects(value):
    assert_valid_json(value)


@pytest.mark.parametrize(
    "value",
    [
        "{",
        '{"a"',
        '{"a":',
        '{"a":1',
        '{"a",1}',
        '{"a" 1}',
        '{"a":}',
        '{:"a"}',
        '{"a":1,}',
        '{"a":1,,',
        '{"a":1]',
        '{"a":[1,2}',
        '{"a":{"b":1}',
    ],
)
def test_invalid_or_incomplete_objects(value):
    assert_invalid_json(value)


# =========================================================
# NESTED JSON
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        '{"a":[1,2,3]}',
        '{"a":[{"b":2}]}',
        '{"a":{"b":[1,2,{"c":3}]}}',
        "[[[1]]]",
        '{"a":{"b":{"c":{"d":null}}}}',
        '[{"a":[{"b":[{"c":true}]}]}]',
        '{"users":[{"name":"A"},{"name":"B"}]}',
    ],
)
def test_valid_nested_json(value):
    assert_valid_json(value)


@pytest.mark.parametrize(
    "value",
    [
        '{"a":[1,2}',
        '{"a":{"b":1]',
        "[[1,2]",
        '{"a":[{"b":1}]',
        '[{"a":1}}]',
        '{"a":{"b":[1,2]}',
    ],
)
def test_invalid_nested_json(value):
    assert_invalid_json(value)


# =========================================================
# EMPTY CONTAINERS
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        "{}",
        "[]",
        "[{}]",
        "[[]]",
        '{"a":{}}',
        '{"a":[]}',
        '{"a":[{}]}',
        '{"a":{"b":[]}}',
    ],
)
def test_empty_containers(value):
    assert_valid_json(value)


# =========================================================
# KEY / VALUE TYPES
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        '{"":1}',
        '{"name":"Hayat"}',
        '{"number":42}',
        '{"bool":true}',
        '{"nothing":null}',
        '{"array":[1]}',
        '{"object":{"x":1}}',
    ],
)
def test_valid_key_value_types(value):
    assert_valid_json(value)


@pytest.mark.parametrize(
    "value",
    [
        "{1:2}",
        "{true:1}",
        "{null:1}",
        "{[]:1}",
        "{{}:1}",
    ],
)
def test_non_string_object_keys_are_invalid(value):
    assert_invalid_json(value)


# =========================================================
# MISMATCHED BRACKETS
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        "[}",
        "{]",
        '{"a":1]',
        "[1}",
        '{"a":[1,2}}',
        '[{"a":1]]',
    ],
)
def test_mismatched_brackets(value):
    assert_invalid_json(value)


# =========================================================
# EXTRA JSON AFTER COMPLETION
# =========================================================


@pytest.mark.parametrize(
    "value",
    [
        "{}{}",
        "[][]",
        '"a""b"',
        "truefalse",
        '{"a":1}{"b":2}',
    ],
)
def test_extra_json_after_complete_value(value):
    assert_invalid_json(value)


# =========================================================
# EXACT ERROR LOCATION
# =========================================================


@pytest.mark.parametrize(
    ("value", "bad_index"),
    [
        ("01", 1),
        ("1..2", 2),
        ("1e--5", 3),
        ("1e+-5", 3),
        ("1e5-6", 3),
        ('"\\x"', 2),
        ("[1}", 2),
        ('{"a":1]', 6),
    ],
)
def test_rejects_at_exact_invalid_character(value, bad_index):
    assert_invalid_at(value, bad_index)


def test_validate_candidate_accepts_valid_continuation():
    machine = JasonStateMacheine()

    assert machine.validate_candidate("true") is True


def test_validate_candidate_rejects_invalid_continuation():
    machine = JasonStateMacheine()

    assert machine.validate_candidate("truX") is False


def test_validate_candidate_does_not_mutate_machine():
    machine = JasonStateMacheine()
    original_state = machine.state
    original_brackets = machine.brackets.copy()

    machine.validate_candidate('{"a":[1,2]}')

    assert machine.state == original_state
    assert machine.brackets == original_brackets


def test_validate_candidate_from_partial_json():
    machine = JasonStateMacheine()

    for char in '{"a":':
        assert machine.transition(char) is True

    assert machine.validate_candidate("true") is True
    assert machine.validate_candidate("}") is False
    assert machine.validate_candidate("[1]") is True
