*This activity has been created as part of the 42 curriculum by hahel.*

# Call Me Maybe

**Hayat Ahel · 42 Irbid**

## Description

Call Me Maybe is a Python command-line tool that turns plain-language requests into structured function calls. For example, if I ask it to multiply 6 by 7, it returns the name of the multiplication function and the two arguments as JSON. The program produces a description of the call for another program to use.

The available functions and their parameter types come from a JSON file at runtime. I used Qwen3-0.6B through 42's supplied `llm_sdk` to choose a function and extract its arguments. I wrote the constrained decoder that controls which tokens the model can select.

## How I worked

I started with one request and a real model response I could inspect. Once I could follow that request from the input file to the generated text, I added JSON grammar rules, then rules for the function definitions, and finally the batch runner and output file. At each stage I ran a small example to see what the model actually produced before checking the behavior with tests.

I kept the prompt builder, JSON grammar, and function-schema checks separate. The prompt gives Qwen the request and the available functions. The grammar handles details like quotes and commas. The schema determines which function and parameter names are allowed and what types their values must have. I kept the original request from the input so I could include it verbatim in the result.

## Constrained decoding

1. I build the model prompt from the request and the runtime function definitions, then tokenize it with the SDK.
2. For each generation step, I ask the model for logits, which are scores for the next token. I consider candidate tokens from highest to lowest score.
3. I check the full text contributed by each candidate, character by character, against the current JSON grammar and function schema. A single token can contain part of a string, a closing quote, and punctuation, so every character needs to be checked. The trial leaves the live state untouched.
4. I accept the first candidate that passes both checks and advance the live state. This picks the same next token as masking every invalid logit and taking the highest remaining score. Each request starts with fresh state.

When the schema already determines a piece of output, such as a fixed key, I write it directly and skip the model call for that text. Once the decoder has selected a function, it can use a shorter, function-specific prompt for the remaining arguments.

The schema constrains the shape and types of the call. Qwen still has to interpret which function the user wants and what values belong in its parameters.

## Challenges faced

- **Getting the right shape.** Early runs returned valid JSON values such as `-2` and `1`. To get an object with `prompt`, `name`, and `parameters`, I added schema checks alongside the JSON grammar and applied them as Qwen generated the call.
- **Making token checks faster.** In an early run, masking took about 1.1–1.25 seconds per token. A later development run measured about 0.13 seconds per token. Checking candidates in score order lets the decoder stop looking as soon as a valid token is found.
- **Preserving punctuation and escapes.** I had to account for how the model's vocabulary represents spaces and how JSON escapes quotes and backslashes. In the graded run, the program handled a Windows path and a template with quoted text. I kept the original request separate from the model's instructions so JSON escaping could preserve its contents.
- **Extracting the exact argument.** In one private test, Qwen picked the right function and returned `"system database"` as an argument. The grader expected `"system"`. The call passed the structural checks because the schema only required a string at that position.

## Testing and performance

I tested the grammar and schema checks directly, then ran the full CLI on 42's public and private datasets and graded the output with the moulinette. One development run passed 469 tests. The focused tests covered JSON numbers, literals, nesting, completion, Unicode handling, and valid parameter-name prefixes. The full runs checked whether the model chose the expected function and extracted the expected arguments.

| Dataset | Correct calls | Accuracy |
| --- | ---: | ---: |
| Public | 11/11 | 100% |
| Private | 10/11 | 90.9% |

In the private run, the CLI saved all 11 records with zero generation failures in 214.03 seconds, about 3 minutes and 34 seconds. All 11 were parseable. The argument mismatch above accounts for the one missed case.

## Instructions

From the repository root, install the dependencies with `uv`. Use the `llm_sdk` supplied with the 42 project and place it alongside `src/` if it is missing from your checkout. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
```

For the default run, put the function definitions in `data/input/functions_definition.json` and the requests in `data/input/function_calling_tests.json`. The CLI writes a JSON array to `data/output/function_calling_results.json`.

## Example usage

```bash
uv run python -m src
```

To supply your own definitions and requests, pass their paths explicitly.

```bash
uv run python -m src \
  --functions_definition examples/functions.json \
  --input examples/requests.json \
  --output data/output/my_results.json
```

With a function named `fn_multiply_numbers` that takes numeric arguments `a` and `b`, an input entry of `{"prompt": "What is the product of 6 and 7?"}` would produce a record like this.

```json
[
  {
    "prompt": "What is the product of 6 and 7?",
    "name": "fn_multiply_numbers",
    "parameters": {
      "a": 6.0,
      "b": 7.0
    }
  }
]
```

## Resources

- The JSON specification, including numbers, strings, and escaping. [RFC 8259](https://www.rfc-editor.org/info/rfc8259/)
- The model used for function selection and argument extraction. [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- Background reading on constraining generation with grammars. [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702)

### AI usage

I wrote the implementation. I used ChatGPT to plan the stages, work through tokenization and constrained decoding, review my code, debug failing cases, and draft this README. Qwen3-0.6B is the model the program uses at runtime.
