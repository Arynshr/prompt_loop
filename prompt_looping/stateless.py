import os
import re
import logging
from functools import lru_cache
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import TypedDict, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

try:
    import tiktoken
    _ENCODER = tiktoken.encoding_for_model("gpt-4")
except Exception:
    _ENCODER = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stateless_agent")

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


class CustomContext(TypedDict):
    user_name: str


class GuardrailError(Exception):
    pass


def estimate_tokens(text: str) -> int:
    if _ENCODER:
        return len(_ENCODER.encode(text))
    return len(text.split())


def validate_input(text: str, cfg: AgentConfig) -> None:
    if not text or not text.strip():
        raise GuardrailError("Empty input rejected")

    token_count = estimate_tokens(text)
    if token_count > cfg.max_input_tokens:
        raise GuardrailError(f"Input exceeds {cfg.max_input_tokens} token limit")

    lowered = text.lower()
    if any(re.search(rf"\b{term}\b", lowered) for term in cfg.banned_terms):
        raise GuardrailError("Input contains disallowed content")


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.get("user_name", "there")
    return (
        f"You are a precise, safety-conscious assistant. "
        f"Address the user as {user_name}. "
        f"Keep responses concise and only use tools when necessary. "
        f"Only call tools if explicitly required."
    )


def _with_truncated_messages(request, truncated):
    if isinstance(request, dict):
        return {**request, "messages": truncated}
    if hasattr(request, "model_copy"):      
        return request.model_copy(update={"messages": truncated})
    if hasattr(request, "copy"):             
        try:
            return request.copy(update={"messages": truncated})
        except TypeError:
            pass
    request.messages = truncated            
    return request


def make_token_limit_middleware(cfg: AgentConfig):
    @before_model
    def enforce_token_limit(request: ModelRequest, runtime=None) -> Optional[ModelRequest]:
        messages = request["messages"] if isinstance(request, dict) else request.messages
        total_tokens = sum(estimate_tokens(str(m.content)) for m in messages)

        if total_tokens > cfg.max_input_tokens:
            logger.warning("Token budget exceeded (%d tokens) - truncating history", total_tokens)
            return _with_truncated_messages(request, messages[-4:])

        return request

    return enforce_token_limit


def get_weather(city: str) -> dict:
    return {"city": city, "weather": "sunny"}


@lru_cache(maxsize=8)
def build_agent(cfg: AgentConfig = AgentConfig()):
    llm = ChatGroq(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
    )
    return create_agent(
        model=llm,
        tools=[get_weather],
        middleware=[dynamic_system_prompt, make_token_limit_middleware(cfg)],
        context_schema=CustomContext,
    )


def run_once(user_input: str, user_name: str, cfg: AgentConfig = AgentConfig()) -> str:
    try:
        validate_input(user_input, cfg)
        agent = build_agent(cfg) 

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            context=CustomContext(user_name=user_name),
        )

        final_msg = result["messages"][-1]
        return final_msg.content

    except GuardrailError as e:
        logger.error("Guardrail blocked request: %s", e)
        return f"Request blocked: {e}"

    except Exception as e:
        logger.exception("Unexpected agent failure: %s", e)
        return f"Sorry, something went wrong: {e}"

class LoopState(TypedDict):
    task: str
    draft: str
    critique: str
    iteration: int


GENERATE_PROMPT = ChatPromptTemplate.from_template(
    "Write a detailed answer for:\n{task}"
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
    """Cached per distinct config, same rationale as build_agent."""
    return ChatGroq(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_output_tokens)


def _safe_invoke(chain, inputs: dict, step_name: str) -> Optional[str]:
    """Run one chain step; never let a single failure crash the whole loop."""
    try:
        return chain.invoke(inputs).content
    except Exception as e:
        logger.exception("Loop step '%s' failed: %s", step_name, e)
        return None


def stateless_loop(task: str, max_iters: int = 3, cfg: AgentConfig = AgentConfig()) -> str:

    validate_input(task, cfg) 

    llm = build_llm(cfg)
    generate_chain = GENERATE_PROMPT | llm
    critique_chain = CRITIQUE_PROMPT | llm
    refine_chain = REFINE_PROMPT | llm

    state: LoopState = {"task": task, "draft": "", "critique": "", "iteration": 0}

    draft = _safe_invoke(generate_chain, {"task": task}, "generate")
    if draft is None:
        return "Sorry, could not generate an initial answer. Please try again."
    state["draft"] = draft

    while state["iteration"] < max_iters:
        state["iteration"] += 1
        prev = state["draft"]

        # Token-limit guardrail: cap draft size before feeding it back into the model
        if estimate_tokens(state["draft"]) > cfg.max_input_tokens:
            logger.warning(
                "Draft exceeds token budget at iteration %d - truncating before critique",
                state["iteration"],
            )
            state["draft"] = state["draft"][: cfg.max_input_tokens * 4]  # rough char-based cap

        critique = _safe_invoke(critique_chain, {"draft": state["draft"]}, f"critique(iter={state['iteration']})")
        if critique is None:
            logger.warning("Critique step failed at iteration %d - returning last good draft", state["iteration"])
            break
        state["critique"] = critique

        refined = _safe_invoke(
            refine_chain,
            {"draft": state["draft"], "critique": state["critique"]},
            f"refine(iter={state['iteration']})",
        )
        if refined is None:
            logger.warning("Refine step failed at iteration %d - returning last good draft", state["iteration"])
            break
        state["draft"] = refined

        if state["draft"].strip() == prev.strip():
            logger.info("Converged early at iteration %d", state["iteration"])
            break

        logger.info("Iteration %d complete", state["iteration"])

    return state["draft"]


def run_loop(task: str, max_iters: int = 3, cfg: AgentConfig = AgentConfig()) -> str:
    """Public entry point: same guardrail/error-handling contract as run_once."""
    try:
        return stateless_loop(task, max_iters=max_iters, cfg=cfg)
    except GuardrailError as e:
        logger.error("Guardrail blocked request: %s", e)
        return f"Request blocked: {e}"
    except Exception as e:
        logger.exception("Unexpected loop failure: %s", e)
        return f"Sorry, something went wrong: {e}"


if __name__ == "__main__":
    print(run_loop("Explain why rate limiting matters for public APIs", max_iters=3))
