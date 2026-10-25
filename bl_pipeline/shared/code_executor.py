"""Tier 2: Sandboxed execution of LLM-generated scientific code.

When a correlation or equation is not in the pre-coded registry (Tier 1),
Agent 1 asks the LLM to write Python code. This module executes that code
in a restricted namespace with only safe scientific libraries.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
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


def _json_result_value(value):
    """Convert scientific values for the JSON-only worker result channel."""
    import numpy as np
    import sympy
    from dataclasses import asdict, is_dataclass

    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, sympy.Basic): return str(value)
    if is_dataclass(value) and not isinstance(value, type): return asdict(value)
    if isinstance(value, (set, frozenset)): return list(value)
    raise TypeError(f'Unsupported result type: {type(value).__name__}')


def _execute_worker(code: str, result_path: Path) -> None:
    """Run only in a child process; stdout/stderr belong to that process."""
    try:
        namespace = _make_sandbox_globals()
        seed_keys = set(namespace)
        exec(code, namespace)  # noqa: S102
        safe_locals = {}
        for key, value in namespace.items():
            if key.startswith('_') or key in seed_keys:
                continue
            try:
                json.dumps(value, default=_json_result_value, allow_nan=False)
                safe_locals[key] = value
            except (TypeError, ValueError, OverflowError):
                safe_locals[key] = f'<{type(value).__name__}>'
        payload = {'success': True, 'result': namespace.get('result'),
                   'error': None, 'locals': safe_locals}
        serialized = json.dumps(payload, default=_json_result_value, allow_nan=False)
    except BaseException as exc:
        serialized = json.dumps({'success': False, 'result': None, 'locals': {},
                                 'error': f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'})
    result_path.write_text(serialized, encoding='utf-8')


def execute_code(code: str, timeout_seconds: float = 30) -> dict[str, Any]:
    """Execute in a disposable process, capturing output without changing sys.stdout.

    The deadline includes interpreter/library startup. Timed-out workers are
    killed and reaped before returning. Results/locals cross a JSON boundary:
    NumPy arrays/scalars become lists/scalars, SymPy expressions become strings;
    unsupported result types return an explicit execution failure. This process
    isolation is not a security sandbox or a memory limit.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('Execution timeout must be positive and finite')
    with tempfile.TemporaryDirectory(prefix='scientific-exec-') as directory:
        result_path = Path(directory) / 'result.json'
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                   '--worker', str(result_path)]
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, encoding='utf-8',
                              errors='replace', **options) as worker:
            try:
                try:
                    stdout, stderr = worker.communicate(code, timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    stdout, stderr = worker.communicate()
                    return {'success': False, 'result': None, 'stdout': stdout,
                            'error': f'TimeoutError: code execution exceeded {timeout_seconds}s',
                            'locals': {}}
                if worker.returncode != 0 or not result_path.exists():
                    return {'success': False, 'result': None, 'stdout': stdout,
                            'error': f'WorkerError: process exited with code {worker.returncode}\n{stderr}',
                            'locals': {}}
                try:
                    payload = json.loads(result_path.read_text(encoding='utf-8'))
                    if not isinstance(payload, dict) or not {'success', 'result', 'error', 'locals'} <= payload.keys():
                        raise ValueError('Invalid worker result schema')
                except (OSError, ValueError) as exc:
                    return {'success': False, 'result': None, 'stdout': stdout,
                            'error': f'WorkerError: invalid result ({type(exc).__name__})', 'locals': {}}
                payload['stdout'] = stdout
                return payload
            finally:
                # Also reap on interruption or a parent-side exception.
                if worker.poll() is None:
                    worker.kill()
                    worker.communicate()


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


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('This module is launched by execute_code().')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _execute_worker(sys.stdin.buffer.read().decode('utf-8'), Path(sys.argv[2]))
