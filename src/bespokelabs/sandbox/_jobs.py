"""Detached jobs: backend lookup and the shared wait loop.

A job runs one command to completion without the submitting process staying
attached, like an AWS Batch job.  Only a backend with a native job service
(``gcp_batch``) runs jobs; it implements :class:`SandboxJobBackend` itself.
Asking any other backend for a job raises :class:`FeatureNotSupportedError`.

The wait loop rides out transient provider read failures: a few consecutive
failures are tolerated before the job has been seen alive; once seen alive,
failures are tolerated until the caller's deadline.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

from bespokelabs.sandbox.exceptions import (
    BackendNotInstalledError,
    FeatureNotSupportedError,
    SandboxAuthenticationError,
    SandboxConfigurationError,
    SandboxTimeoutError,
)
from bespokelabs.sandbox.types import JobStatus

logger = logging.getLogger(__name__)

# Wait-loop tolerance: consecutive failed provider reads tolerated before the
# job has been seen alive, and the ceiling on the backoff between them.
_PROVIDER_READ_ATTEMPTS = 5
_PROVIDER_READ_MAX_BACKOFF = 60.0

# Errors no amount of retrying fixes; the wait loop raises them at once.
_FATAL_ERRORS = (
    SandboxConfigurationError,
    SandboxAuthenticationError,
    BackendNotInstalledError,
    FeatureNotSupportedError,
)


def check_supports_jobs(backend_name: str, backend_cls: type, op: str) -> None:
    """Raise unless *backend_cls* runs jobs natively.

    Checked on the class, so it needs no backend SDK installed (and a mock
    instance does not qualify by accident).
    """
    if not callable(getattr(backend_cls, "submit_job", None)):
        raise FeatureNotSupportedError(
            f"The '{backend_name}' backend does not support jobs; "
            "use SandboxClient('gcp_batch')",
            backend=backend_name,
            op=op,
        )


def job_runner(backend_name: str, backend_client: Any, op: str) -> Any:
    """Return the backend client if it runs jobs natively, else raise."""
    check_supports_jobs(backend_name, type(backend_client), op)
    return backend_client


# -- Wait loop -----------------------------------------------------------------


class _Waiter:
    """Polling policy shared by the sync and async ``wait_job``."""

    def __init__(
        self, job_id: str, timeout_secs: float | None, poll_secs: float
    ) -> None:
        if timeout_secs is not None and timeout_secs < 0:
            raise SandboxConfigurationError("timeout_secs must be >= 0")
        self._job_id = job_id
        self._timeout = timeout_secs
        self._deadline = (
            None if timeout_secs is None else time.monotonic() + timeout_secs
        )
        self._poll = max(0.1, float(poll_secs))
        self._failures = 0
        self._observed = False
        self._last: JobStatus | None = None

    def _remaining(self) -> float | None:
        if self._deadline is None:
            return None
        return self._deadline - time.monotonic()

    def _timeout_error(self) -> SandboxTimeoutError:
        state = self._last.state.value if self._last else "unobserved"
        return SandboxTimeoutError(
            f"Timed out after {self._timeout}s waiting for the job; "
            f"last state: {state}",
            op="wait_job",
            context={"job_id": self._job_id, "state": state},
        )

    def observed(self, status: JobStatus) -> float | None:
        """Record a status; the delay before the next poll, or ``None`` if done."""
        self._failures = 0
        self._last = status
        if status.done:
            return None
        self._observed = True
        remaining = self._remaining()
        if remaining is not None and remaining <= 0:
            raise self._timeout_error()
        return self._poll if remaining is None else min(self._poll, remaining)

    def failed(self, exc: Exception) -> float:
        """Record a failed read; the delay before retrying, or raise."""
        if isinstance(exc, _FATAL_ERRORS):
            raise exc
        self._failures += 1
        remaining = self._remaining()
        if remaining is not None and remaining <= 0:
            if self._observed:
                raise self._timeout_error() from exc
            raise exc
        if self._failures >= _PROVIDER_READ_ATTEMPTS and not self._observed:
            raise exc
        logger.info(
            "Job status read failed (%d in a row); retrying: %s",
            self._failures,
            exc,
        )
        index = min(self._failures, _PROVIDER_READ_ATTEMPTS) - 1
        delay = min(_PROVIDER_READ_MAX_BACKOFF, self._poll * 2**index)
        return delay if remaining is None else max(0.0, min(delay, remaining))


def wait_sync(
    status_fn: Callable[[str], JobStatus],
    job_id: str,
    *,
    timeout_secs: float | None,
    poll_secs: float,
    on_poll: Callable[[JobStatus], Any] | None,
) -> JobStatus:
    waiter = _Waiter(job_id, timeout_secs, poll_secs)
    while True:
        try:
            status = status_fn(job_id)
        except Exception as exc:
            time.sleep(waiter.failed(exc))
            continue
        if on_poll is not None:
            on_poll(status)
        delay = waiter.observed(status)
        if delay is None:
            return status
        time.sleep(delay)


async def wait_async(
    status_fn: Callable[[str], JobStatus],
    job_id: str,
    *,
    timeout_secs: float | None,
    poll_secs: float,
    on_poll: Callable[[JobStatus], Any] | None,
) -> JobStatus:
    waiter = _Waiter(job_id, timeout_secs, poll_secs)
    while True:
        try:
            status = await asyncio.to_thread(status_fn, job_id)
        except Exception as exc:
            await asyncio.sleep(waiter.failed(exc))
            continue
        if on_poll is not None:
            outcome = on_poll(status)
            if inspect.isawaitable(outcome):
                await outcome
        delay = waiter.observed(status)
        if delay is None:
            return status
        await asyncio.sleep(delay)
