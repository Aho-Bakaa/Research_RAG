"""langfuse_tracer.py — Production tracing with graceful offline no-op fallback.

Implements section 7 and 9.3 of RAG_ARCHITECTURE.md:
- Traces end-to-end spans: routing -> candidate pool -> MMR -> ReAct reasoning.
- Transparent no-op fallback when Langfuse credentials are not set in environment.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Generator

log = logging.getLogger(__name__)

_LANGFUSE_CLIENT = None


def get_langfuse_client() -> Any | None:
    """Initialize Langfuse client if configured in environment."""
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not pub_key or not sec_key:
        return None

    try:
        from langfuse import Langfuse
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        _LANGFUSE_CLIENT = Langfuse(public_key=pub_key, secret_key=sec_key, host=host)
        return _LANGFUSE_CLIENT
    except Exception as e:
        log.warning(f"Failed to initialize Langfuse client: {e}")
        return None


@contextlib.contextmanager
def trace_span(
    name: str,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> Generator[Any | None, None, None]:
    """Context manager for tracing a hierarchical span. Degrades to no-op offline."""
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        span = client.span(name=name, metadata=metadata or {}, trace_id=trace_id)
        yield span
    except Exception as e:
        log.debug(f"Langfuse trace error: {e}")
        yield None
    finally:
        pass
