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


