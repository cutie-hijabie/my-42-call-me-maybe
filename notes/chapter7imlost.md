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






Yes. If each block gets its **own chat**, we need a handoff-quality master checklist so each chat has a strict job and doesn't start redesigning the other blocks.

I'm also locking in the decision we just reached: **outer keys use candidate-key constrained decoding**. We are not relying on prompting alone, and we are not doing semantic mappings like `greet → fn_greet`.

# Chapter 7 — Semantic Function Calling

## Locked architecture

The generated call ultimately has this required shape:

```text
{
    "prompt": <original user request>,
    "name": <LLM-selected runtime function>,
    "parameters": <arguments extracted by LLM>
}
```

The subject requires the original prompt, selected function name, and parameters in the output. :chatgpt-content-reference{index="0"}

Responsibilities stay separated:

```text
JSON STATE MACHINE
→ Is this valid JSON?

SCHEMA STATE / MASKING
→ Is this token compatible with the expected schema?

QWEN
→ Which legal token has the highest probability?
→ Which function semantically matches?
→ What argument values are in the request?

RUNTIME FUNCTION DEFINITIONS
→ What functions exist?
→ What parameters do they require?
→ What types do those parameters have?
```

And the rule throughout Chapter 7:

> **The mask restricts possibilities. Qwen chooses among legal possibilities.**

The function itself must be selected using the LLM, not keyword rules or other semantic heuristics. :chatgpt-content-reference{index="1"}

---

# 🟦 BLOCK 1 — PROMPT

### Goal

Correctly generate/validate the first part of the function-call object and reach the NAME block with the state machines synchronized.

### A. Outer candidate-key system

- [ ] Establish the legal outer candidate keys:
  - `prompt`
  - `name`
  - `parameters`
- [ ] Decide where those candidate keys live so we don't scatter literals through `masking.py`.
- [ ] Add state for the **partial outer key** if tokenization requires it.
- [ ] Restrict key tokens by candidate-prefix matching.
- [ ] Correctly recognize when a candidate key has been completed.
- [ ] Prevent invalid/unknown outer keys.
- [ ] Prevent already-completed outer keys from being generated again if applicable.

This should follow the same general prefix principle already used for runtime parameter names. :chatgpt-content-reference{index="2"}

### B. Entering the prompt field

- [ ] Correctly identify that the generated outer key corresponds to the PROMPT block.
- [ ] Allow JSON punctuation/structure to continue normally.
- [ ] Don't duplicate responsibilities already handled by `JasonStateMacheine`.

### C. Prompt value

- [ ] Decide precisely what the constraint on the prompt value should be.
- [ ] Handle a prompt string split across multiple tokenizer tokens.
- [ ] Handle spaces correctly.
- [ ] Handle punctuation correctly.
- [ ] Handle quotes/escapes/special characters correctly.
- [ ] Ensure the generated prompt corresponds to the original request as required by the subject.
- [ ] Do this from the **runtime user prompt**, never from a test-specific value such as `"Greet shrek"`.

### D. Leaving PROMPT

- [ ] Detect completion of the prompt value.
- [ ] Handle the following JSON structure correctly.
- [ ] Transition `SchemaState` from `PROMPT` → `NAME` at the correct moment.
- [ ] Confirm JSON state and schema state remain synchronized.

Your existing state machine already contains a PROMPT → NAME transition, so we should modify/reuse it rather than create a second competing mechanism. :chatgpt-content-reference{index="3"}

### E. PROMPT tests

My job in that chat:

- [ ] Candidate-key prefix tests.
- [ ] Invalid outer-key test.
- [ ] Multi-token key test if tokenizer behavior requires it.
- [ ] Prompt-value tests.
- [ ] Multi-token prompt test.
- [ ] Special-character prompt test.
- [ ] State-transition test.
- [ ] Regression tests against existing JSON behavior.
- [ ] Real Qwen `"Greet shrek"` diagnostic.

### BLOCK 1 EXIT CONDITION

We **do not move on** just because one test passes.

PROMPT is complete when we can confidently say:

> The constrained decoder can enter the required object, handle the outer candidate-key mechanism, produce the prompt portion correctly from runtime input, and reliably arrive at the NAME block.

Then **STOP. New chat.**

---

# 🟪 BLOCK 2 — NAME

### Goal

Qwen must semantically choose the correct function from whatever functions are supplied at runtime.

This is the most important semantic block.

### A. Enter NAME

- [ ] Confirm PROMPT leaves us in the correct JSON/schema state.
- [ ] Outer candidate-key machinery handles the `name` field correctly.
- [ ] Enter NAME value generation cleanly.

### B. Runtime function candidates

The candidates **must come from the supplied definitions**, not from our five sample functions.

- [ ] Obtain current runtime function names.
- [ ] Preserve `partial_function_name`.
- [ ] Candidate-prefix matching works across token boundaries.
- [ ] Exact-match handling works correctly.
- [ ] Invalid function names become impossible.
- [ ] Valid runtime function names remain possible.

A lot of this machinery already exists. :chatgpt-content-reference{index="4"}

**Do not rewrite working Chapter 6 behavior unnecessarily.**

### C. Semantic selection

- [ ] Confirm Qwen actually has enough semantic context to distinguish functions.
- [ ] No keyword → function mappings.
- [ ] No regex-based semantic function selection.
- [ ] No assumptions about the five provided function names.
- [ ] Test descriptions/signatures only based on measured accuracy if prompt tuning is needed.
- [ ] Verify Qwen's logits—not Python logic—determine which surviving function wins.

### D. Function-selection testing

Use supplied examples such as:

```text
Greet shrek
→ fn_greet

What is the sum of 2 and 3?
→ fn_add_numbers

Reverse the string 'hello'
→ fn_reverse_string
```

but also:

- [ ] Different names.
- [ ] Similar function names.
- [ ] Functions with overlapping purposes.
- [ ] Completely different runtime function sets.
- [ ] Reordered definitions.
- [ ] Hidden-test-style definitions.

The subject explicitly allows evaluation with different prompts and function sets, so our implementation cannot depend on the provided five. :chatgpt-content-reference{index="5"}

### E. Leave NAME

- [ ] Store selected function.
- [ ] Make its definition available to PARAMETERS.
- [ ] Transition NAME → PARAMETERS correctly.
- [ ] Preserve JSON/schema synchronization.

### BLOCK 2 EXIT CONDITION

NAME is finished when:

> Given runtime function definitions and a natural-language request, Qwen selects among only valid runtime function names, without semantic heuristics, and the selected function reliably determines the schema used by PARAMETERS.

Then **STOP. New chat.**

---

# 🟩 BLOCK 3 — PARAMETERS

### Goal

Given the function Qwen selected, generate **all and only** its required parameters with correct names, types, and values.

### A. Load selected function schema

- [ ] Retrieve the selected function's parameters dynamically.
- [ ] No assumptions about `a`, `b`, `name`, `s`, etc.
- [ ] Handle different numbers of parameters.
- [ ] Handle parameter order appropriately.
- [ ] Handle zero parameters if allowed by the definitions/subject.

### B. Parameter candidate keys

We already have much of this.

- [ ] Get candidate parameter names from selected runtime definition.
- [ ] Track partial parameter names.
- [ ] Prefix-match candidate keys.
- [ ] Detect completed keys.
- [ ] Track `completed_parameters`.
- [ ] Prevent duplicates.
- [ ] Prevent unknown parameters.
- [ ] Know when all required parameters have been generated.

Existing code already contains parameter-key candidate filtering, so again: **extend/fix, don't casually rewrite it.** :chatgpt-content-reference{index="6"}

### C. Parameter types

- [ ] Retrieve expected type dynamically.
- [ ] String parameter → JSON string.
- [ ] Number parameter → JSON number.
- [ ] Preserve generic JSON validation.
- [ ] Wrong JSON types are impossible.
- [ ] Check any other types the subject/runtime definitions actually require before adding support for them.

Existing masking already has string/number type handling. :chatgpt-content-reference{index="7"}

### D. Semantic value extraction

This is where Qwen does real work.

For:

```text
What is the sum of 265 and 345?
```

Qwen must extract:

```text
a → 265
b → 345
```

The decoder can enforce:

```text
a is required
b is required
both are numbers
```

but **must not contain a special rule that extracts `265` and `345`.**

Tests:

- [ ] strings
- [ ] numbers
- [ ] multiple parameters
- [ ] large numbers
- [ ] empty strings
- [ ] spaces
- [ ] apostrophes/quotes
- [ ] special characters
- [ ] regex strings
- [ ] repeated words
- [ ] parameter values appearing in unusual positions

### E. Finish generation

- [ ] All required parameters present.
- [ ] No extra parameters.
- [ ] No duplicates.
- [ ] Correct closing `}`.
- [ ] Correct closing outer object.
- [ ] `JasonStateMacheine` reaches DONE correctly.
- [ ] generation stops correctly.
- [ ] resulting text always parses as JSON.
- [ ] resulting object satisfies required schema.

### BLOCK 3 EXIT CONDITION

PARAMETERS is finished when:

> For a dynamically selected function, the decoder restricts generation to that function's parameter schema while Qwen extracts the actual argument values from the user's request.

Then **STOP.**

---

# 🔴 BLOCK 4 — CHAPTER 7 INTEGRATION

I want a fourth chat after the three blocks. **Do not call Chapter 7 finished after PARAMETERS.**

This chat is where we put all three together.

### Full pipeline

```text
user request
      ↓
semantic model context
      ↓
PROMPT
      ↓
NAME
      ↓
PARAMETERS
      ↓
complete valid function call
```

Checklist:

- [ ] Run all 11 supplied prompts.
- [ ] Automatically compare expected function selection.
- [ ] Automatically compare extracted arguments.
- [ ] Calculate actual accuracy.
- [ ] Target **≥90% semantic accuracy**. :chatgpt-content-reference{index="8"}
- [ ] Verify **100% valid JSON**.
- [ ] Verify **100% schema-valid output**.
- [ ] Test changed runtime function definitions.
- [ ] Test reordered functions.
- [ ] Test unfamiliar function names.
- [ ] Test edge cases.
- [ ] Test malformed inputs/error handling where applicable.
- [ ] Run **every existing test**, not merely Chapter 7 tests.
- [ ] Fix regressions.
- [ ] Check generation performance.
- [ ] Confirm we aren't using private `llm_sdk` internals.
- [ ] Confirm no prohibited semantic heuristics slipped in.
- [ ] Only then mark Chapter 7 complete.

---

## The rule for all four chats

I want us to keep asking one question whenever we're unsure:

> **Is this information STRUCTURE, a RUNTIME CONSTRAINT, or a SEMANTIC DECISION?**

```text
STRUCTURE
→ JSON/schema machinery

RUNTIME CONSTRAINT
→ definitions / candidate sets / types

SEMANTIC DECISION
→ Qwen
```

If we catch ourselves making Python decide something that belongs in the third category, **stop**.

And if we catch ourselves asking Qwen to magically guarantee something that belongs in the first category, **stop**.

That distinction is the backbone of Chapter 7.

**Current position:**

> 🟦 **BLOCK 1 — PROMPT → A. Outer candidate-key system**

That's exactly where the new chat should begin.
