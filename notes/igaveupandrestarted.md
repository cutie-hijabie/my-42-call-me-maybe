1. No existing place/function currently loads the input JSON files. This will need to be connected through the entry point (__main__.py).

2. build_prompt() takes one user request + the runtime function definitions and returns a text prompt containing the context for the model.

3. wrapper_encode() converts the model prompt string into token IDs because the model operates on token IDs, not raw text.

4. wrapper_get_logits_from_input_ids() gives a score (logit) for every possible next token. The higher the logit, the more the model prefers that token as the next token.

5. After invalid tokens are masked, generation uses argmax to select the token with the highest valid logit as the next token.

6. After selecting a token, json_state and schema_state are updated with what was actually generated so they know the current position/state and can constrain the next token correctly.

7. The selected token is appended to token_ids. On the next loop, the model receives the original prompt tokens + everything generated so far, allowing it to predict the next token.

8. Generation stops when either the EOS token is generated, the maximum length is reached, or the JSON state machine reports that the JSON structure is complete.

9. generate_constrained() currently returns the token IDs for BOTH the original model prompt and the generated response, not just the newly generated output.

10. wrapper_decode() converts token IDs back into readable text. We eventually want to decode only the newly generated portion, not the original model prompt.

11. Most existing modules are reusable. prompt.py, schema_state.py, and masking.py are implemented but still incomplete for the final project. The major missing piece is the entry-point/orchestration layer: nothing currently loads the runtime input files and connects them to the existing pipeline through to readable output.

12. Entry point owns orchestration. prompt.py owns model context construction. generation.py owns token generation. JSON/schema/masking own constraints. LLMWrapper owns communication with the SDK.

```markdown
## Intended Journey

```text
PROGRAM STARTS
      ↓
load function_calling_tests.json
load functions_definition.json
      ↓
validate/parse function definitions using schema.py
      ↓
take request
e.g. "Greet shrek"
      ↓
build_prompt(request, definitions)
      ↓
generate_constrained(...)
      ↓
isolate newly generated token IDs
      ↓
decode generated IDs
      ↓
readable generated response
```

---

## Shared Setup vs Per-Request Work

### Once When the Program Starts

```text
create LLMWrapper
      ↓
build vocab_map
      ↓
load function definitions
      ↓
load requests
```

These are things that can be shared across all of the requests.

### For Each Request

```text
take one request
      ↓
build its prompt
      ↓
create fresh SchemaState
      ↓
generate
      ↓
isolate generated tokens
      ↓
decode
      ↓
store / display result
```

Each request gets its own generation process and fresh state.

1. json.load() gives us raw Python data. InputFileDefinition.model_validate() validates and converts the function definitions into our Pydantic schema. .root gives us the actual List[InputDefinition] used by the rest of the program.

2. Performance profiling: model logits take roughly 0.2–0.3s/token, while masking rises to roughly 1.1–1.25s/token. Masking is currently the dominant generation bottleneck and needs optimization.

3. Masking optimization succeeded: masking dropped from ~1.1–1.25 s/token to ~0.13 s/token. Masking is no longer the dominant bottleneck.

4. Generation still does not terminate quickly. Next investigation: generation stopping conditions — starting prompt length, max_length semantics, and whether json_state.is_complete() becomes true.

5. Starting model prompt is 77 tokens, so the current 500-token limit is not already exceeded.

6. After 60+ generated tokens, json_state.is_complete() remained false. The next diagnostic is generated-token content, not further performance profiling.

7. Masking was externally optimized for performance; before accepting that change, verify that the optimization preserved constraint behavior.

8. Opening-quote state bug: SchemaState can record the opening JSON quote as part of partial_key_candidate because the JSON machine transitions to IN_STRING/key before the schema state processes that character. JSON syntax and actual key content are therefore being mixed.

9. Empty-valid-token-set bug: generation does not detect when masking leaves zero valid tokens. max() still selects an index when every logit is -inf, potentially selecting token ID 0 and continuing generation with a token that was actually masked.

10. Constraint fix: opening " is no longer stored as part of partial_key_candidate.

11. Generation guard: if masking leaves zero valid next tokens, generation now raises an explicit error instead of allowing max() to silently choose token ID 0. This does not solve constraint bugs itself; it makes them observable.

12. Prompt-value constraint defect: outer-key generation now correctly reaches {"prompt":". Immediately after entering the prompt string value, masking leaves zero valid next tokens. The failure is therefore now localized to constraints/state handling for the "prompt" value, rather than outer-key generation.

13. Integration result: real requests now travel through entry point → prompt construction → Qwen → constrained decoding → readable decoded output.

14. Baseline performance: observed completed simple requests around 5.5–7 seconds after masking optimization.

15. Current implementation boundary: generation beyond the prompt field reaches partially implemented name/parameters behavior. Failures there should be handled when those schema stages are built/completed, rather than mistaken for an entry-point failure.

16. Safety improvement: zero-valid-token situations now fail explicitly instead of silently degenerating into token-ID-0 generation.

17. __JSON syntax and function-call structure are separate.__ `{}` and `{"prompt":"..."}` are complete, valid JSON. They still fail the project’s output contract because required fields are missing.

18. __A token can contain multiple characters and cross several JSON states.__ Candidate validation must simulate every character in order; checking only the token’s first character is insufficient.

19. __Candidate validity does not require completion.__ A token that leaves an unfinished string or object can be a valid continuation. Completion is required when returning the final response.

20. __Candidate simulation must preserve live state.__ `validate_candidate()` copies the machine and separately copies its mutable bracket list. Rejected candidates must not alter subsequent candidate checks or actual generation.

21. __Number-whitespace bug:__ accepting whitespace without leaving `IN_NUMBER` allowed invalid input such as `[1 2]`. The space was ignored, letting the next digit continue the previous number.

22. __Number-whitespace fix:__ whitespace now ends a complete number, moving to `AFTER_VALUE` inside a container or `DONE` at the root. If `need_digit` is true, whitespace is rejected because the number is unfinished.

23. __Literal-whitespace bug:__ the general whitespace shortcut also accepted invalid fragments such as `t rue`, `fal se`, and `nu ll`. Whitespace is now rejected during `IN_LITERAL`; whitespace after a completed literal remains legal.

24. __Number-state leakage:__ `has_exponent_sign` from a previous number could block a new negative number, such as in `[1e+2, -3]`. Number-start branches now accept `-` independently; exponent-sign restrictions remain inside `IN_NUMBER`.

25. __Unicode escape masking bug:__ after `\u`, the machine remains in `IN_STRING_ESCAPE` while expecting four hexadecimal digits. The escape-prefix shortcut incorrectly excluded valid digit candidates. That shortcut now applies only when `digits_needed == 0`, leaving pending hexadecimal validation to the grammar.

26. __Generated-token limits exclude model input.__ `len(token_ids) - input_length` counts generated tokens. The instruction prompt’s tokens do not consume the current generation allowance.

27. __Stopping and successful completion are different.__ Reaching the token limit or selecting EOS can stop the loop while JSON is unfinished. Stopping alone must not cause a normal return.

28. __Completion guard:__ after the loop, generation raises `ValueError` if `jason.is_complete()` is false. Complete JSON still succeeds when its final token lands exactly at the limit.

29. __Test import-path lesson:__ `uv run python -m pytest` makes the current project directory available for imports such as `src.json_state`. `pytest -m` has a different purpose: selecting test markers.

30. __Unsaved-code diagnosis:__ three completion tests initially reported “DID NOT RAISE.” The guard was correct but unsaved. Saving the file made all eight generation tests pass.

31. __Grammar verification:__ the existing JSON-state suite and 29 new regression cases passed together: __248 tests passed__. Coverage included whitespace boundaries, exponent-state leakage, Unicode masking, and candidate-state preservation.

32. __Generation verification:__ __8 focused cases passed__, using controlled fake-model scores with real generation and masking. They checked selection, multi-character advancement, fresh state, completion, truncation, no legal candidates, and premature EOS.

33. __Real-model observation:__ the updated run returned __7 parseable JSON responses__—3 prompt-only objects and 4 empty objects—and reported __4 explicit generation failures__. No malformed or unfinished response was returned in that run.

34. __Remaining diagnostic boundary:__ “No valid next token” means all candidates were excluded. Because masking combines JSON and schema rules, the reported JSON state alone does not establish which rule caused the failure.

35. __Next stage:__ enforce complete records with exactly `prompt`, `name`, and `parameters`, runtime-valid function names, required49. `ParameterDefinition.type` now uses `Literal["string", "number", "integer", "boolean", "null"]`. This validates supported type names in runtime definitions; masking separately constrains generated values.

36. Value-start constraints: strings start with `"`, numbers and integers with a digit or `-`, booleans with `t` or `f`, and null with `n`. JSON grammar validates the remaining syntax.

37. Integers also need continuation constraints. A temporary JSON-state copy checks candidate characters and rejects `.`, `e`, and `E` while reading the integer, including across token boundaries. Live state remains unchanged.

38. Whitespace-only tokens are legal before values. The string-start check needs a `copy` guard so an empty stripped candidate is not rejected.

39. Masking tests must account for both JSON grammar and schema constraints. An array can be valid JSON while being invalid as our function-call record. Tests now use actual `SchemaState` and runtime definitions.

40. __Chapter 3.1 verification:__ __379 selected tests passed__, covering runtime contracts, scalar constraints, and regressions. Complete record enforcement remains Chapter 3.2–3.3 work.

41. `outer_key_candidates` tracks unused keys. Removing a key after its closing quote does not mean its value has finished.

42. `required_outer_keys` is an immutable snapshot of the required keys. `completed_outer_fields` separately tracks fields whose values have finished.

43. Outer-field completion is recorded after JSON advancement: `prompt`/`name` finish on a closing quote with `AFTER_VALUE` at bracket depth 1; `parameters` finishes on its closing brace with the same state and depth.

44. `has_complete_outer_fields()` compares completed fields with required keys. It does not yet establish that every required parameter is present or that generation should stop.

45. __Current position:__ Level 3 → Chapter 3.2 → Step 1 → Substep 1.1. Completion tracking has been added; next we connect it to candidate masking. Chapter 3.2 formal testing remains at its end. parameters, matching types, and exact original-request preservation. Structurally permitted function calls can still have incorrect meaning.

46. Outer-field progression: candidate checks enforce prompt → name → parameters, including tokens crossing key, colon, and value boundaries.

47. Original-request preservation: prompt matching uses its JSON-serialized form, including quotes, escapes, control characters, and Unicode.

48. Runtime function selection: names remain candidates until the closing quote completes an exact runtime name. Shared prefixes do not select a function prematurely.

49. State synchronization: candidate simulation preserves live states. Real generation refreshes the selected function’s parameters for every character.

50. Chapter 3.2 verification: 545 tests passed in 6.09 seconds.

51. Real-model observation: 10 complete records and 1 explicit failure. Successful generation times total 22 minutes 47 seconds, exceeding the five-minute whole-batch limit. The Shrek failure’s cause remains unresolved; the vowel replacement also showed a semantic error.
