import json
import time
import sys
from pydantic import ValidationError
from .generation import generate_constrained
from .llm_wrapper import LLMWrapper
from .vocab import build_vocab_map
from .prompt import build_prompt
from .schema_state import SchemaState
import os
from contextlib import redirect_stderr
from pathlib import Path
import argparse
from .schema import InputFileDefinition, InputFilePrompts


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Translate requests into structured function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the requests JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path for the generated results JSON file.",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()
    results: list[dict[str, object]] = []
    failed_requests: int = 0
    try:
        with open(args.input, "r", encoding="utf-8") as file:
            raw_prompt_data = json.load(file)

        prompt_data = InputFilePrompts.model_validate(raw_prompt_data).root

    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
    ) as error:
        print(
            f"Could not load requests from '{args.input}': {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        with open(args.functions_definition, "r", encoding="utf-8") as file:
            definition_data = json.load(file)

        validated_data = InputFileDefinition.model_validate(definition_data)
        definitions = validated_data.root

    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValidationError,
    ) as error:
        print(
            "Could not load function definitions from "
            f"'{args.functions_definition}': {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as quiet_stream:
            with redirect_stderr(quiet_stream):
                llm_model = LLMWrapper()
                vocab = build_vocab_map(llm_model)

    except (OSError, ValueError, RuntimeError) as error:
        print(
            f"Could not initialize the model or vocabulary: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    function_names = [definition.name for definition in definitions]
    no_match_marker = "__no_match__"
    while no_match_marker in function_names:
        no_match_marker += "_"

    for index, line in enumerate(prompt_data, start=1):
        start = time.perf_counter()
        try:
            og_prompt = line.prompt
            model_prompt = build_prompt(
                og_prompt, definitions, no_match_marker
            )

            prompt_schema = SchemaState(og_prompt)

            generated = generate_constrained(
                model_prompt,
                500,
                llm_model,
                vocab,
                prompt_schema,
                function_names,
                definitions,
                no_match_marker=no_match_marker,
            )

            elapsed = time.perf_counter() - start

            model_prompt_token_ids = llm_model.wrapper_encode(model_prompt)
            generated_output_ids = generated[len(model_prompt_token_ids):]
            result = json.loads(llm_model.wrapper_decode(generated_output_ids))
            results.append(result)

            generated_output_readable = json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )

            elapsed = time.perf_counter() - start

            print(
                f"Request {index}/{len(prompt_data)}: {elapsed:.2f}s\n"
                f"Output: {generated_output_readable}\n"
            )
        except (OSError, ValueError, RuntimeError) as error:
            failed_requests += 1
            print(
                f"Request {index}/{len(prompt_data)} failed "
                f"for {line.prompt!r}: {error}",
                file=sys.stderr,
            )

    output_path = Path(args.output)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                results,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

    except (OSError, ValueError) as error:
        print(
            f"Failed to save results to '{output_path}': {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    total_elapsed = time.perf_counter() - total_start

    print(
        f"Requests: {len(prompt_data)} | "
        f"Saved: {len(results)} | "
        f"Failed: {failed_requests} | "
        f"Total time: {total_elapsed:.2f}s"
    )

    if failed_requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
