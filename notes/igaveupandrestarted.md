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
