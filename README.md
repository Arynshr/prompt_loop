# Prompt Looping

> LLM-level loop -> the user holds the agent over iterative prompts; the user holds the autonomy over the responses generated throughout the course of the loop.

- You repeatedly call the model with updated prompts
- The loop is driven by **prompt → response → new prompt**
- Usually controlled manually or via a framework

### Key characteristics:

- Can be stateful/stateless, depending on whether the user defines memory
- Scripted by the user, runs iteratively
- Autonomy remains with the author of the prompts (not the LLM itself)
- Static loop mechanism
- In practice, wrapped with guardrails (input validation, banned-content checks), a token budget, and per-step error handling so one bad LLM call doesn't crash the whole loop

## Prompt control loop

Deterministic loop controlling the LLM's response; the response of the LLM depends on its previous response.

> Constantly evolving prompts with repeated LLM calls

```python
state = initial_input()
while not done:
    prompt = build_prompt(state)
    output = llm(prompt)
    state = update(state, output)
```

#### Without memory

- Stateless across iterations - responses are non-persistent and cannot be reused as a prompt to the LLM once the call returns
- Context -> only what is passed as input by the user (explicitly), per call
- Deterministic and predictable
- State managed entirely by the caller, rebuilt fresh on every invocation

```python
state = {prev_response}
```

**In practice** (see `stateless_agent.py`):

- Each call validates input, builds its own chain/agent from a cached LLM instance, and runs generate → critique → refine with no shared mutable state across calls
- A convergence guard exits early when a refinement stops changing the draft
- A token check truncates the draft before feeding it back in, since it can grow every iteration
- Failures are caught per step (`_safe_invoke`) so the loop returns the last good draft instead of crashing

#### With Memory

- Memory introduced as an abstraction for response persistence
- Memory externalizes state management from your loop

```python
LangChain manages:
    memory.store(messages)
    memory.inject(prompt)
```

**In practice** (see `stateful_agent.py`):

- A `MemoryStore` interface (`get` / `save` / `clear`) keyed by `session_id` decouples the loop from _where_ state lives - in-memory dict for process lifetime, JSON file (or a DB/Redis store) for persistence across restarts
- On each call, prior state is loaded; if present, the prompt **evolves** the previous draft (`EVOLVE_PROMPT`) instead of regenerating from scratch
- Iteration count is cumulative across calls for a session, so `max_iters` bounds new refinement work per call, not the session's whole history
- Stored draft size is capped independently from the live per-call token limit, so memory doesn't grow unbounded across many sessions

### Iterative prompt loop

> Linear refinement -> each iteration enhances the response by transforming/critiquing the previous response Generate -> Critique -> Improve -> Simplify (linear flow)

- Deterministic / static loop
- Fixed number of steps, or convergence-based exit
- Cons -> context growth may be non-linear (mitigated with a token check + truncation per iteration)

### Recursive prompt loop

> Loop calls itself with modified inputs

```python
def f(x):
    if base_case(x):
        return result
    sub_results = [f(x1), f(x2), f(x3), ...]
    return combine(sub_results)
```

- Non-linear structure of outputs
- Cons -> context overflow
