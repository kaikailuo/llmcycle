from __future__ import annotations

import socket
import ssl
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

_RETRYABLE_TRANSPORT_ERRORS = (
    TimeoutError,
    socket.timeout,
    ConnectionResetError,
    ConnectionAbortedError,
    ssl.SSLError,
)


def _contains_certificate_verification_error(error: BaseException) -> bool:
    current: object = error
    seen: set[int] = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        if isinstance(current, URLError):
            current = current.reason
        else:
            current = current.__cause__ or current.__context__
    return False


def _is_retryable(error: BaseException) -> bool:
    # HTTPError is a URLError subclass, so status handling must come first.
    if isinstance(error, HTTPError):
        return error.code in RETRYABLE_HTTP_STATUS_CODES
    if _contains_certificate_verification_error(error):
        return False
    if isinstance(error, URLError):
        return True
    return isinstance(error, _RETRYABLE_TRANSPORT_ERRORS)


def urlopen_with_retry(
    request: Any,
    *,
    timeout: float,
    opener: Callable[..., Any] = urlopen,
    max_attempts: int = 3,
    backoff_seconds: tuple[float, ...] = (1.0, 2.0),
    sleeper: Callable[[float], Any] = time.sleep,
) -> Any:
    """Open one HTTP request, retrying only transient transport failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(max_attempts):
        try:
            return opener(request, timeout=timeout)
        except Exception as exc:
            if not _is_retryable(exc) or attempt == max_attempts - 1:
                raise
            if backoff_seconds:
                delay_index = min(attempt, len(backoff_seconds) - 1)
                sleeper(backoff_seconds[delay_index])

    raise AssertionError("retry loop exited unexpectedly")
