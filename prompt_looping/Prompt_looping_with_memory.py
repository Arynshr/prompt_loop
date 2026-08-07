import os
import re
import logging
from functools import lru_cache
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import TypedDict, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from prompt_registry import LoopState, MemoryStore, InMemoryStore, JSONFileStore, DEFAULT_STORE

try:
    import tiktoken
    _ENCODER = tiktoken.encoding_for_model("gpt-4")
except Exception:
    _ENCODER = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stateful_agent")

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise EnvironmentError("Groq API key not available in environment")


@dataclass(frozen=True)
class AgentConfig:
    model: str = "llama-3.1-8b-instant"
    max_input_tokens: int = 2000
    max_output_tokens: int = 1024
    temperature: float = 0.4
    banned_terms: tuple = ("hack", "exploit", "malware")
    max_history_chars: int = 6000  # cap on stored draft size, independent of live token limit


class GuardrailError(Exception):
    pass


def estimate_tokens(text: str) -> int:
    if _ENCODER:
        return len(_ENCODER.encode(text))
    return len(text.split())


def validate_input(text: str, cfg: AgentConfig) -> None:
    if not text or not text.strip():
        raise GuardrailError("Empty input rejected")
    if estimate_tokens(text) > cfg.max_input_tokens:
        raise GuardrailError(f"Input exceeds {cfg.max_input_tokens} token limit")
    lowered = text.lower()
    if any(re.search(rf"\b{term}\b", lowered) for term in cfg.banned_terms):
        raise GuardrailError("Input contains disallowed content")

# Memory: state persists ACROSS calls, keyed by session_id.
# Schema (LoopState) and storage backends now live in prompt_registry.py

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    "Write a detailed answer for:\n{task}"
)
EVOLVE_PROMPT = ChatPromptTemplate.from_template(
    "You previously produced this answer in an earlier session:\n{previous_draft}\n\n"
    "New/follow-up task:\n{task}\n\n"
    "Write an updated answer that builds on the previous one where it's still relevant, "
    "rather than starting from scratch."
)
CRITIQUE_PROMPT = ChatPromptTemplate.from_template(
    "You are a strict reviewer. List weaknesses only.\n\nAnswer:\n{draft}"
)
REFINE_PROMPT = ChatPromptTemplate.from_template(
    "Improve the answer using ONLY the critique. Do NOT rewrite from scratch.\n\n"
    "Answer:\n{draft}\n\nCritique:\n{critique}"
)


@lru_cache(maxsize=8)
def build_llm(cfg: AgentConfig = AgentConfig()) -> ChatGroq:
    return ChatGroq(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_output_tokens)


def _safe_invoke(chain, inputs: dict, step_name: str) -> Optional[str]:
    try:
        return chain.invoke(inputs).content
    except Exception as e:
        logger.exception("Loop step '%s' failed: %s", step_name, e)
        return None



# Stateful loop

def stateful_loop(
    session_id: str,
    task: str,
    max_iters: int = 3,
    cfg: AgentConfig = AgentConfig(),
    store: Optional[MemoryStore] = None,
) -> str:
    store = store or DEFAULT_STORE
    validate_input(task, cfg)

    llm = build_llm(cfg)
    generate_chain = GENERATE_PROMPT | llm
    evolve_chain = EVOLVE_PROMPT | llm
    critique_chain = CRITIQUE_PROMPT | llm
    refine_chain = REFINE_PROMPT | llm

    prior_state = store.get(session_id)

    if prior_state and prior_state.get("draft"):
        logger.info("Session %s: evolving from memory (prior iteration=%d)", session_id, prior_state["iteration"])
        draft = _safe_invoke(
            evolve_chain,
            {"previous_draft": prior_state["draft"], "task": task},
            "evolve",
        )
        start_iteration = prior_state["iteration"]
    else:
        logger.info("Session %s: no memory found - starting fresh", session_id)
        draft = _safe_invoke(generate_chain, {"task": task}, "generate")
        start_iteration = 0

    if draft is None:
        return "Sorry, could not generate an answer. Please try again."

    state: LoopState = {
        "session_id": session_id,
        "task": task,
        "draft": draft,
        "critique": "",
        "iteration": start_iteration,
    }

    loop_target = start_iteration + max_iters
    while state["iteration"] < loop_target:
        state["iteration"] += 1
        prev = state["draft"]

        if estimate_tokens(state["draft"]) > cfg.max_input_tokens:
            logger.warning("Draft exceeds token budget at iteration %d - truncating", state["iteration"])
            state["draft"] = state["draft"][: cfg.max_input_tokens * 4]

        critique = _safe_invoke(critique_chain, {"draft": state["draft"]}, f"critique(iter={state['iteration']})")
        if critique is None:
            logger.warning("Critique step failed at iteration %d - stopping", state["iteration"])
            break
        state["critique"] = critique

        refined = _safe_invoke(
            refine_chain,
            {"draft": state["draft"], "critique": state["critique"]},
            f"refine(iter={state['iteration']})",
        )
        if refined is None:
            logger.warning("Refine step failed at iteration %d - stopping", state["iteration"])
            break
        state["draft"] = refined

        if state["draft"].strip() == prev.strip():
            logger.info("Converged early at iteration %d", state["iteration"])
            break

        logger.info("Session %s: iteration %d complete", session_id, state["iteration"])

    # Persist evolved state for the next call, capping stored size independent of live token checks
    persisted = dict(state)
    persisted["draft"] = state["draft"][: cfg.max_history_chars]
    store.save(session_id, persisted)

    return state["draft"]


def run_stateful_loop(
    session_id: str,
    task: str,
    max_iters: int = 3,
    cfg: AgentConfig = AgentConfig(),
    store: Optional[MemoryStore] = None,
) -> str:
    try:
        return stateful_loop(session_id, task, max_iters=max_iters, cfg=cfg, store=store)
    except GuardrailError as e:
        logger.error("Guardrail blocked request: %s", e)
        return f"Request blocked: {e}"
    except Exception as e:
        logger.exception("Unexpected loop failure: %s", e)
        return f"Sorry, something went wrong: {e}"


if __name__ == "__main__":
    session = "demo-session-1"
    print(run_stateful_loop(session, "Explain why rate limiting matters for public APIs"))
    print("---")
    # Second call, same session: evolves the previous answer instead of starting over
    print(run_stateful_loop(session, "Now extend it to cover rate limiting for internal microservices too"))
    print("---")
    print(f"Registry sessions on record: {DEFAULT_STORE.list_sessions()}")
