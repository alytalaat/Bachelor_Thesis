"""
llm_manager.py — Multi-key Groq LLM manager with round-robin rotation.

Supports 5 Groq API keys. On every call the next key in the cycle is used.
On a 429 rate-limit error the manager immediately tries the next key instead
of waiting, cycling through all keys before falling back to a timed wait.

Drop-in replacement: invoke_llm and invoke_verifier_llm signatures unchanged.
New export:          invoke_coder_llm  — used by agent.py for the coder role.
New export:          set_models        — swap planner/coder/verifier at runtime
                                         for benchmark ablation runs.
"""

import os
import time
import itertools
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────────────
# Load all 5 keys. Keys not present in .env are silently skipped.
_RAW_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
]
GROQ_API_KEYS: list[str] = [k for k in _RAW_KEYS if k]

if not GROQ_API_KEYS:
    # Fall back to the single legacy key so existing .env files keep working
    legacy = os.getenv("GROQ_API_KEY")
    if legacy:
        GROQ_API_KEYS = [legacy]
    else:
        raise EnvironmentError(
            "No Groq API keys found. Set GROQ_API_KEY_1 … GROQ_API_KEY_5 "
            "(or GROQ_API_KEY for a single key) in your .env file."
        )

print(f"[LLM Manager] {len(GROQ_API_KEYS)} Groq API key(s) loaded — round-robin rotation enabled.")

# ── Key cycling ───────────────────────────────────────────────────────────────
# Three independent cycles — planner, verifier, and coder each rotate through
# all keys separately so they never compete for the same key at the same time.
_planner_cycle  = itertools.cycle(GROQ_API_KEYS)
_verifier_cycle = itertools.cycle(GROQ_API_KEYS)
_coder_cycle    = itertools.cycle(GROQ_API_KEYS)

# ── Default model names ───────────────────────────────────────────────────────
# Change these by calling set_models() before a benchmark run.
_PLANNER_MODEL  = "openai/gpt-oss-120b"
_VERIFIER_MODEL = "llama-3.3-70b-versatile"
_CODER_MODEL    = "qwen/qwen3-32b"

_PLANNER_TEMP  = 0
_VERIFIER_TEMP = 0
_CODER_TEMP    = 0.7


def set_models(
    planner:  str | None = None,
    verifier: str | None = None,
    coder:    str | None = None,
) -> None:
    """
    Swap model names at runtime without restarting the process.
    Pass only the roles you want to change; omit the rest.

    Example — ablation run with Llama as planner:
        set_models(planner="llama-3.3-70b-versatile")

    Example — ablation run with Gemma as coder:
        set_models(coder="gemma2-9b-it")
    """
    global _PLANNER_MODEL, _VERIFIER_MODEL, _CODER_MODEL
    if planner  is not None: _PLANNER_MODEL  = planner
    if verifier is not None: _VERIFIER_MODEL = verifier
    if coder    is not None: _CODER_MODEL    = coder
    print(
        f"[LLM Manager] Models updated — "
        f"Planner: {_PLANNER_MODEL} | "
        f"Verifier: {_VERIFIER_MODEL} | "
        f"Coder: {_CODER_MODEL}"
    )


def get_models() -> dict:
    """Returns the currently active model names."""
    return {
        "planner":  _PLANNER_MODEL,
        "verifier": _VERIFIER_MODEL,
        "coder":    _CODER_MODEL,
    }


# ── Core invoke helper ────────────────────────────────────────────────────────

def _invoke_with_rotation(
    messages:    list,
    model:       str,
    temperature: float,
    key_cycle:   itertools.cycle,
    role_label:  str,
    extra_kwargs: dict | None = None,
) -> object:
    """
    Invokes a ChatGroq model using round-robin key rotation.

    On each call:
      1. Pick the next key from the cycle.
      2. If a 429 is returned, immediately try the next key.
      3. After exhausting all keys once, wait 10 s and retry from the top.
      4. Hard-fails after 3 full rotations (~15 keys × 3 = 45 attempts max).
    """
    extra_kwargs = extra_kwargs or {}
    num_keys     = len(GROQ_API_KEYS)
    max_rotations = 3

    for rotation in range(max_rotations):
        for attempt in range(num_keys):
            api_key = next(key_cycle)
            try:
                client = ChatGroq(
                    model=model,
                    temperature=temperature,
                    api_key=api_key,
                    **extra_kwargs,
                )
                return client.invoke(messages)

            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "rate" in err.lower() or "rate_limit" in err.lower()
                is_auth_error = "401" in err or "invalid api key" in err.lower()

                if is_auth_error:
                    print(f"[LLM {role_label}] Auth error on key …{api_key[-6:]} — skipping key permanently.")
                    # Remove bad key from pool so it is never tried again
                    if api_key in GROQ_API_KEYS:
                        GROQ_API_KEYS.remove(api_key)
                    continue

                if is_rate_limit:
                    print(
                        f"[LLM {role_label}] 429 on key …{api_key[-6:]} "
                        f"(rotation {rotation + 1}/{max_rotations}, "
                        f"attempt {attempt + 1}/{num_keys}) — trying next key..."
                    )
                    continue  # immediately try next key

                # Non-rate-limit error — re-raise immediately
                raise

        # Exhausted all keys in this rotation — wait before next rotation
        wait = 10 * (rotation + 1)
        print(
            f"[LLM {role_label}] All {num_keys} keys rate-limited "
            f"(rotation {rotation + 1}/{max_rotations}) — waiting {wait}s..."
        )
        time.sleep(wait)

    raise RuntimeError(
        f"[LLM {role_label}] Rate-limited on all {num_keys} keys "
        f"after {max_rotations} full rotations. Increase key count or wait longer."
    )


# ── Public invoke functions ───────────────────────────────────────────────────

def invoke_llm(messages: list) -> object:
    """
    Coordinator / Planner LLM call.
    Uses _PLANNER_MODEL with round-robin key rotation.
    """
    extra = {}
    if "gpt-oss" in _PLANNER_MODEL or "reasoning" in _PLANNER_MODEL:
        extra = {"reasoning_effort": "high"}
    return _invoke_with_rotation(
        messages=messages,
        model=_PLANNER_MODEL,
        temperature=_PLANNER_TEMP,
        key_cycle=_planner_cycle,
        role_label="Planner",
        extra_kwargs=extra,
    )


def invoke_verifier_llm(messages: list) -> object:
    """
    Verifier / Check-3 LLM call.
    Uses _VERIFIER_MODEL with round-robin key rotation.
    Verifier is kept fixed during ablation runs — only planner and coder vary.
    """
    return _invoke_with_rotation(
        messages=messages,
        model=_VERIFIER_MODEL,
        temperature=_VERIFIER_TEMP,
        key_cycle=_verifier_cycle,
        role_label="Verifier",
    )


def invoke_coder_llm(messages: list) -> object:
    """
    Coder LLM call (user_coder and admin_coder).
    Uses _CODER_MODEL with round-robin key rotation.

    NOTE: agent.py instantiates llm_coder = ChatGroq(...) directly.
    Replace that line with:
        from llm_manager import invoke_coder_llm as _coder_invoke
    and call _coder_invoke(messages) instead of llm_coder.invoke(messages).
    Or keep llm_coder and monkey-patch it — see MIGRATION NOTE below.
    """
    return _invoke_with_rotation(
        messages=messages,
        model=_CODER_MODEL,
        temperature=_CODER_TEMP,
        key_cycle=_coder_cycle,
        role_label="Coder",
    )


# ── Convenience: patch the coder in agent.py without editing it ───────────────
# agent.py calls:  response = llm_coder.invoke(coder_prompt)
# If you import this module before agent.py runs, you can monkey-patch like so:
#
#   import llm_manager
#   import agent
#   agent.llm_coder = llm_manager._CoderProxy()
#
# _CoderProxy wraps invoke_coder_llm behind a .invoke() method so agent.py
# needs zero changes.

class _CoderProxy:
    """Thin wrapper so agent.py's llm_coder.invoke() calls invoke_coder_llm."""
    def invoke(self, messages):
        return invoke_coder_llm(messages)

    # LangChain chains sometimes call __or__ for piping — passthrough
    def __or__(self, other):
        return other


# Export a ready-made proxy instance
coder_proxy = _CoderProxy()