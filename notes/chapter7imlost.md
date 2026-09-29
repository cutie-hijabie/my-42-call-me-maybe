### What `build_prompt()` needs to accomplish

Forget the exact Python for a moment. The function has **four jobs**:

**Task 1 — Start with a tiny instruction.**

Tell Qwen what decision it's making. Something short like the idea you already had:

```text
Choose the function that matches the request and fill its parameters.
```

We can experiment with this wording later for accuracy.

**Task 2 — Give Qwen the available functions dynamically.**

This is where your loop comes in.

```python
for definition in definitions:
```

The purpose of that loop is:

> Take **each function that exists in the current runtime file** and add enough information about it to the prompt for Qwen to distinguish it from the others.

It doesn't care whether there are 5 functions, 20 functions, or completely different functions during review.

For each iteration, `definition` represents **one function**.

So conceptually:

```text
iteration 1 → information about function A
iteration 2 → information about function B
iteration 3 → information about function C
...
```

That's why there's no hardcoding.

And yes, we're going to experiment with **how much information** from each definition Qwen actually needs. The subject tells us the definition file contains names, argument names/types, return type, and descriptions, but it does **not** say you must dump all of that into the LLM prompt. :chatgpt-content-reference{index="2"}

So your peer's experience is relevant. We can test **name + parameters**, **name + short description + parameters**, etc., and measure which gives Qwen better accuracy instead of assuming "more context = better."

**Task 3 — Add the actual user's request.**

After the loop has described the choices, append:

```text
Request: Greet shrek
```

That comes from `user_prompt`, obviously—not hardcoded.

**Task 4 — Return the finished string.**

That's the string we'll eventually encode instead of encoding raw `"Greet shrek"`.

So the overall transformation is simply:

```text
user_prompt + runtime definitions
              ↓
          build_prompt()
              ↓
short instruction
available function A
available function B
available function C
...
user request
```

Then Qwen gets **that**.

Chapter 7 — Semantic Function Calling Checklist
- [x] 1. Establish the failure baseline. "Greet shrek" with the old raw prompt generated -2, proving valid JSON alone isn't enough.
- [x] 2. Build runtime semantic context. build_prompt() now dynamically gives Qwen an instruction, available function signatures, and the user's request.
- [x] 3. Unit-test build_prompt(). Passed. We know Qwen receives the function definitions dynamically rather than hardcoded choices.
- [x] 4. Retry real Qwen generation with semantic context. It generated 1. This exposed that the decoder still permits outputs that are valid JSON but invalid for our required function-call schema.
- [x] 5. Identify the missing layer. SchemaState knows PROMPT → NAME → PARAMETERS, but masking only has meaningful schema constraints for NAME and PARAMETERS; the beginning/full output grammar isn't yet enforced generically.   schema_state
- [ ] 6. Define the complete output grammar/state model. Before coding, map what the decoder must know at every point in { "prompt": ..., "name": ..., "parameters": ... } without prompt-specific/function-specific hardcoding.
- [ ] 7. Implement generic top-level schema constraints. The decoder should derive legal structural/key tokens from the expected output schema/state, rather than a collection of special cases for our current test.
- [ ] 8. Constrain the dynamic prompt value correctly. The generated prompt field must correspond to the current input while remaining part of constrained generation.
- [ ] 9. Verify NAME integration. Once generation reaches the name value, Qwen chooses among the runtime function_names; no keyword mapping or heuristic selection.
- [ ] 10. Verify PARAMETERS integration. Parameter keys/types come dynamically from the function Qwen selected, while Qwen extracts the actual values.
- [ ] 11. Get "Greet shrek" passing end-to-end. Expected full structured object, not merely parseable JSON.
- [ ] 12. Run all 11 supplied prompts. Measure function-selection and argument-extraction accuracy rather than eyeballing individual examples.
- [ ] 13. Stress-test generalization. Different function names/descriptions/parameters, ambiguous prompts, empty strings, special characters, large numbers, etc.—the subject specifically warns that evaluation won't be limited to the supplied examples.     en.subject
- [ ] 14. Tune semantic context empirically. Compare compact signatures vs descriptions/other prompt variants only if measured accuracy requires it. Target is 90%+, not “this prompt sounds good.”     en.subject
- [ ] 15. Full regression. All existing Chapter 1–6 tests + Chapter 7 tests must pass. We don't sacrifice your 414 passing tests to get semantics working.
- [ ] 16. Then move to project integration. CLI/input/output/error handling and final subject-level verification come after the semantic engine works.
