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


14. json.load() gives us raw Python data. InputFileDefinition.model_validate() validates and converts the function definitions into our Pydantic schema. .root gives us the actual List[InputDefinition] used by the rest of the program.

15. Performance profiling: model logits take roughly 0.2–0.3s/token, while masking rises to roughly 1.1–1.25s/token. Masking is currently the dominant generation bottleneck and needs optimization.

16. Masking optimization succeeded: masking dropped from ~1.1–1.25 s/token to ~0.13 s/token. Masking is no longer the dominant bottleneck.

17. Generation still does not terminate quickly. Next investigation: generation stopping conditions — starting prompt length, max_length semantics, and whether json_state.is_complete() becomes true.

18. Starting model prompt is 77 tokens, so the current 500-token limit is not already exceeded.

19. After 60+ generated tokens, json_state.is_complete() remained false. The next diagnostic is generated-token content, not further performance profiling.

20. Masking was externally optimized for performance; before accepting that change, verify that the optimization preserved constraint behavior.

21. Opening-quote state bug: SchemaState can record the opening JSON quote as part of partial_key_candidate because the JSON machine transitions to IN_STRING/key before the schema state processes that character. JSON syntax and actual key content are therefore being mixed.

22. Empty-valid-token-set bug: generation does not detect when masking leaves zero valid tokens. max() still selects an index when every logit is -inf, potentially selecting token ID 0 and continuing generation with a token that was actually masked.

23. Constraint fix: opening " is no longer stored as part of partial_key_candidate.

24. Generation guard: if masking leaves zero valid next tokens, generation now raises an explicit error instead of allowing max() to silently choose token ID 0. This does not solve constraint bugs itself; it makes them observable.

25. Prompt-value constraint defect: outer-key generation now correctly reaches {"prompt":". Immediately after entering the prompt string value, masking leaves zero valid next tokens. The failure is therefore now localized to constraints/state handling for the "prompt" value, rather than outer-key generation.

26. Integration result: real requests now travel through entry point → prompt construction → Qwen → constrained decoding → readable decoded output.

27. Baseline performance: observed completed simple requests around 5.5–7 seconds after masking optimization.

28. Current implementation boundary: generation beyond the prompt field reaches partially implemented name/parameters behavior. Failures there should be handled when those schema stages are built/completed, rather than mistaken for an entry-point failure.

29. Safety improvement: zero-valid-token situations now fail explicitly instead of silently degenerating into token-ID-0 generation.

