from src.json_state import JasonStateMacheine
from src.schema import InputDefinition, ParameterDefinition, ReturnsDefinition
from src.schema_state import SchemaState


def test_schema_state_stays_synced_with_json_machine_for_real_tokens():
    definition = InputDefinition(
        name="fn_greet",
        description="Greet someone",
        parameters={"name": ParameterDefinition(type="string")},
        returns=ReturnsDefinition(type="string"),
    )
    definitions = [definition]
    function_names = [definition.name]

    jason = JasonStateMacheine()
    schema_state = SchemaState()

    tokens = [
        "{",
        '"prompt"',
        ":",
        '"Say hello"',
        ",",
        '"name"',
        ":",
        '"fn_greet"',
        ",",
        '"parameters"',
        ":",
        "{",
        '"name"',
        ":",
        '"Hayat"',
        "}",
        "}",
    ]

    for token_string in tokens:
        for char in token_string:
            jason.transition(char)

        parameters = schema_state.get_parameters(definitions)
        schema_state.update(
            token_string,
            jason,
            function_names,
            parameters,
        )

    assert schema_state.function_name == "fn_greet"
    assert schema_state.current_parameter == "name"
