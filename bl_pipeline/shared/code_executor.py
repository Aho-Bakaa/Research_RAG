"""Tier 2: Sandboxed execution of LLM-generated scientific code.

When a correlation or equation is not in the pre-coded registry (Tier 1),
Agent 1 asks the LLM to write Python code. This module executes that code
in a restricted namespace with only safe scientific libraries.
"""

from __future__ import annotations

import signal
import threading
import traceback
from io import StringIO
from typing import Any


# Allowed builtins in the sandbox.  Kept tight on purpose (no exec/eval/
# compile/open/input/__import__/globals/locals/vars/dir/help/breakpoint)
# but generous on the introspection and iteration primitives that
# scientific code routinely uses.  Bug 1 from run #4 was Sonnet calling
# `type(x)` to diagnose its own sandbox-state confusion, which crashed
# with NameError and burned a turn — that was a stupid restriction.
_SAFE_BUILTINS = {
    # arithmetic / numeric
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "pow": pow, "divmod": divmod,
    # iteration
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "reversed": reversed, "filter": filter, "map": map,
    "all": all, "any": any, "iter": iter, "next": next,
    # type constructors
    "float": float, "int": int, "str": str, "list": list, "dict": dict,
    "tuple": tuple, "set": set, "frozenset": frozenset, "bool": bool,
    "bytes": bytes, "bytearray": bytearray, "complex": complex,
    # constants
    "True": True, "False": False, "None": None,
    # I/O + formatting
    "print": print, "repr": repr, "format": format, "ascii": ascii,
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
    # introspection (Bug 1 fix — these are SAFE and Sonnet uses them
    # routinely for self-diagnostics; blocking them serves no purpose)
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
    "callable": callable, "id": id, "hash": hash,
    # exception classes Sonnet sometimes catches inside compute()
    "Exception": Exception, "ValueError": ValueError,
    "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "ArithmeticError": ArithmeticError,
    "RuntimeError": RuntimeError, "AssertionError": AssertionError,
    "StopIteration": StopIteration,
}


_ALLOWED_MODULES = {
    "math", "numpy", "np", "scipy", "sympy",
    # A1 correlation library — expose for LLM-generated code so the
    # REASONER can call evaluate_ags_eq3(), fs20_eq36_re_tr(), etc.
    # directly instead of re-deriving formulas every run.
    "bl_pipeline.agent1_fresh.correlations",
    "bl_pipeline",  # parent — needed so dotted import resolves
}


def _safe_import(name, *args, **kwargs):
    """Import function that only allows whitelisted scientific modules."""
    if name.split(".")[0] in _ALLOWED_MODULES:
        return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) else __import__(name, *args, **kwargs)
    raise ImportError(f"Import of '{name}' is not allowed in sandbox")


def _make_sandbox_globals() -> dict[str, Any]:
    """Create a restricted global namespace for code execution.

    Pre-imports the A1 correlation library so REASONER-written code can
    call helpers like `ags_re_theta_t(2.8)`, `fs20_eq36_re_tr(0.028, 13,
    0.013, 1.5e-5)`, `x_sweep_forward(...)`, `phase2_back_fit(...)`
    without needing an import statement.  Anchor formulas are sourced
    from the original papers — see `correlations.py` docstrings.
    """
    import math

    import numpy as np
    import scipy
    import sympy

    # A1 correlation library — surface the key public functions in the
    # sandbox namespace so the LLM can use them directly.
    from bl_pipeline.agent1_fresh import correlations as a1_correlations

    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import

    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "math": math,
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "sympy": sympy,
        "pi": math.pi,
        "e": math.e,
        # The full module is also accessible as `correlations.foo` if the
        # LLM prefers that style.
        "correlations": a1_correlations,
    }

    # Expose every public function from correlations.py at the top level —
    # `ags_re_theta_t(2.8)` works directly, no import needed.  Constants
    # and private (_-prefixed) names are intentionally excluded.
    for name in dir(a1_correlations):
        if name.startswith("_"):
            continue
        obj = getattr(a1_correlations, name)
        if callable(obj) or hasattr(obj, "__dataclass_fields__"):
            namespace[name] = obj

    return namespace


def _exec_with_timeout(
    code: str, sandbox_globals: dict[str, Any], timeout_seconds: int,
) -> None:
    """Run exec(code, sandbox_globals), aborting after timeout_seconds.

    Uses SIGALRM on platforms that support it (Unix).  CPython cannot
    forcibly kill a running thread, so on platforms without SIGALRM
    (Windows) this runs exec() on a daemon thread and raises
    TimeoutError if it hasn't finished by the deadline; the runaway
    thread is abandoned (not killed) but, being a daemon, it will not
    block process exit.
    """
    if hasattr(signal, "SIGALRM"):
        def _on_alarm(signum, frame):
            raise TimeoutError(f"code execution exceeded {timeout_seconds}s")

        previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout_seconds)
        try:
            exec(code, sandbox_globals)  # noqa: S102
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
        return

    caught: list[BaseException] = []

    def _run() -> None:
        try:
            exec(code, sandbox_globals)  # noqa: S102
        except BaseException as exc:  # re-raised on the caller's thread below
            caught.append(exc)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(f"code execution exceeded {timeout_seconds}s")
    if caught:
        raise caught[0]


def execute_code(code: str, timeout_seconds: int = 30) -> dict[str, Any]:
    """Execute LLM-generated scientific code in a sandbox.

    Args:
        code: Python source code to execute.
        timeout_seconds: Max wall-clock execution time in seconds, enforced
            via SIGALRM on Unix or a daemon-thread join timeout elsewhere.

    Returns:
        dict with keys:
        - "success": bool
        - "result": the value of the `result` variable if set
        - "stdout": captured print output
        - "error": error message if failed
        - "locals": dict of all local variables after execution
    """
    sandbox_globals = _make_sandbox_globals()

    # Capture stdout
    import sys
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()

    try:
        # Use a SINGLE namespace for both globals and locals.  With two
        # separate dicts, functions/list-comprehensions defined inside
        # the executed code only see `sandbox_globals`, so module-level
        # assignments (x_t = ..., U_inf = ...) become invisible inside
        # helper functions and crash with NameError. Passing one dict
        # makes it behave like a real Python module, which is what the
        # LLM-generated code expects.
        _exec_with_timeout(code, sandbox_globals, timeout_seconds)

        sandbox_locals = sandbox_globals
        result = sandbox_locals.get("result", None)
        stdout_text = captured.getvalue()

        # Filter out non-serialisable objects from locals, and drop the
        # sandbox-provided globals (math, numpy, builtins) so we only
        # report user-bound names.
        _sandbox_seed_keys = set(_make_sandbox_globals().keys())
        safe_locals = {}
        for k, v in sandbox_locals.items():
            if k.startswith("_") or k in _sandbox_seed_keys:
                continue
            try:
                repr(v)  # test if representable
                safe_locals[k] = v
            except Exception:
                safe_locals[k] = str(type(v))

        return {
            "success": True,
            "result": result,
            "stdout": stdout_text,
            "error": None,
            "locals": safe_locals,
        }

    except Exception as exc:
        return {
            "success": False,
            "result": None,
            "stdout": captured.getvalue(),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "locals": {},
        }

    finally:
        sys.stdout = old_stdout


def validate_code_safety(code: str) -> tuple[bool, str]:
    """Basic safety check on LLM-generated code.

    Rejects code that tries to:
    - Import dangerous modules (os, subprocess, sys, etc.)
    - Use file I/O
    - Use exec/eval (nested)
    - Access dunder attributes

    NOTE: getattr(/setattr( are intentionally NOT forbidden here — they are
    whitelisted in _SAFE_BUILTINS (Bug 1 fix, see the comment there) because
    Sonnet uses them routinely for safe self-diagnostics; rejecting them here
    would contradict what the sandbox actually allows. delattr( stays
    forbidden since it is not in _SAFE_BUILTINS.
    """
    forbidden_patterns = [
        "import os",
        "import sys",
        "import subprocess",
        "import shutil",
        "import pathlib",
        "from os",
        "from sys",
        "from subprocess",
        "open(",
        "exec(",
        "eval(",
        "__import__",
        "globals(",
        "delattr(",
        "compile(",
    ]

    code_lower = code.lower()
    for pattern in forbidden_patterns:
        if pattern.lower() in code_lower:
            return False, f"Forbidden pattern detected: {pattern}"

    return True, "OK"
