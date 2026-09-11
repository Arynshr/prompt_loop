# Prompt Looping

> Exploring iterative LLM control loops where model outputs become inputs to subsequent reasoning steps.

**Prompt Looping** is a small experimental project exploring how repeated LLM calls can be structured into controlled refinement loops.

Instead of treating an LLM call as a one-shot operation, the loop maintains a state that evolves through repeated **prompt → response → update** cycles.

## Core Idea

```text
Input
  ↓
Build Prompt
  ↓
LLM
  ↓
Response
  ↓
Update State
  ↓
Converged?
 ├─ No → repeat
 └─ Yes → return result
```

The project explores different ways of managing this loop:

* **Stateless loops** — state is explicitly passed between iterations.
* **Stateful loops** — persistent state is maintained across calls.
* **Iterative refinement** — generate → critique → improve.
* **Recursive loops** — recursively decompose and combine intermediate results.
* **Convergence control** — stop when further iterations no longer improve the result.
* **Guardrails** — token limits, validation, and per-step failure handling.

## Architecture

### Stateless

Each invocation owns its state and constructs the next iteration explicitly.

```text
Input
  ↓
Generate
  ↓
Critique
  ↓
Refine
  ↓
Output
```

No persistent state exists between independent calls.

### Stateful

A memory abstraction persists the evolving state across calls.

```text
Session
  ↓
Load State
  ↓
Evolve Prompt
  ↓
LLM
  ↓
Save State
  ↓
Next Iteration
```

The implementation keeps memory storage separate from the loop itself, allowing the backing store to be replaced without changing the control logic.

## Iterative vs Recursive

### Iterative

```text
Generate → Critique → Improve → Simplify → ...
```

A linear refinement process with a fixed iteration limit or convergence-based termination.

### Recursive

```text
              Task
           /    |    \
         A      B      C
        / \    / \    / \
      ... ... ... ... ... ...
           ↓
        Combine
```

Recursive loops allow a task to be decomposed into multiple independent sub-problems before their results are combined.

## Design Principles

* Keep **loop control deterministic**.
* Keep **state management explicit**.
* Bound iterations and token usage.
* Fail safely and preserve the last valid result.
* Stop when additional refinement provides no meaningful change.
* Keep the orchestration layer independent from the underlying LLM.

## Project Structure

```text
prompt_looping/
├── stateless_agent.py
├── stateful_agent.py
└── ...
```

## Status

Experimental / research prototype.

The project is primarily intended to explore **LLM control-flow patterns, state management, and iterative reasoning architectures** rather than provide a production-ready agent framework.

## License

MIT
