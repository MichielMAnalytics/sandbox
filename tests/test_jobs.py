"""Detached job API tests that need no provider.

Jobs run on ``gcp_batch`` only (see test_gcp_batch_backend.py); these cover
the spec, the shared wait loop, and the error every other backend raises.
"""

from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal
from fractions import Fraction
from unittest import mock

from bespokelabs.sandbox import (
    AsyncSandboxClient,
    JobSpec,
    JobState,
    JobStatus,
    SandboxClient,
    _jobs,
)
from bespokelabs.sandbox.exceptions import (
    BackendNotInstalledError,
    FeatureNotSupportedError,
    SandboxAuthenticationError,
    SandboxConfigurationError,
    SandboxConnectionError,
    SandboxTimeoutError,
)


class JobSpecTests(unittest.TestCase):

    def test_defaults(self) -> None:
        spec = JobSpec(command=["echo", "hi"])
        self.assertEqual(spec.timeout_secs, 3600)
        self.assertEqual(spec.cpu, 1.0)
        self.assertEqual(spec.memory_mb, 1024)
        self.assertEqual(spec.env_vars, {})

    def test_rejects_empty_or_malformed_commands(self) -> None:
        for command in ("", "   ", [], ["ok", 3], 42):
            with self.subTest(command=command):
                with self.assertRaises(SandboxConfigurationError):
                    JobSpec(command=command)  # type: ignore[arg-type]

    def test_tuple_command_becomes_a_list(self) -> None:
        self.assertEqual(JobSpec(command=("a", "b")).command, ["a", "b"])

    def test_rejects_non_positive_resources(self) -> None:
        for field in ("cpu", "memory_mb", "timeout_secs", "disk_mb"):
            for value in (0, -1, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(SandboxConfigurationError):
                        JobSpec(command="true", **{field: value})

    def test_rejects_fractional_memory_and_timeout(self) -> None:
        for field in ("memory_mb", "timeout_secs"):
            for value in (
                0.5,
                1024.5,
                Decimal("1024.5"),
                Fraction(3, 2),
                float("inf"),
                Decimal("Infinity"),
            ):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(SandboxConfigurationError):
                        JobSpec(command="true", **{field: value})
        spec = JobSpec(command="true", memory_mb=2048.0, timeout_secs=60.0)
        self.assertEqual((spec.memory_mb, spec.timeout_secs), (2048.0, 60.0))

    def test_rejects_infinite_or_oversized_resources(self) -> None:
        for field in ("cpu", "memory_mb", "timeout_secs", "disk_mb"):
            for value in (float("inf"), 10**400):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(SandboxConfigurationError):
                        JobSpec(command="true", **{field: value})

    def test_rejects_env_names_the_shell_cannot_export(self) -> None:
        for name in ("1A", "A-B", "A B", ""):
            with self.subTest(name=name):
                with self.assertRaises(SandboxConfigurationError):
                    JobSpec(command="true", env_vars={name: "x"})
        with self.assertRaises(SandboxConfigurationError):
            JobSpec(command="true", env_vars={"A": 1})  # type: ignore[dict-item]

    def test_status_done(self) -> None:
        for state in JobState:
            status = JobStatus(job_id="j", state=state)
            self.assertEqual(
                status.done,
                state not in (JobState.PENDING, JobState.RUNNING),
            )


class UnsupportedBackendTests(unittest.TestCase):
    """Backends without a native job service refuse every job method."""

    JOB = "projects/p/locations/us-central1/jobs/job-1"

    def _calls(self, client: SandboxClient) -> dict:
        return {
            "submit_job": lambda: client.submit_job(JobSpec(command="true")),
            "job_status": lambda: client.job_status(self.JOB),
            "wait_job": lambda: client.wait_job(self.JOB, timeout_secs=1),
            "job_logs": lambda: client.job_logs(self.JOB),
            "cancel_job": lambda: client.cancel_job(self.JOB),
        }

    def test_sync_job_methods_raise_not_supported(self) -> None:
        client = SandboxClient("local")
        for op, call in self._calls(client).items():
            with self.subTest(op=op):
                with self.assertRaises(FeatureNotSupportedError) as ctx:
                    call()
                self.assertIn("does not support jobs", str(ctx.exception))
                self.assertEqual(ctx.exception.backend, "local")
                self.assertEqual(ctx.exception.op, op)


class AsyncUnsupportedBackendTests(unittest.IsolatedAsyncioTestCase):

    async def test_async_job_methods_raise_not_supported(self) -> None:
        client = AsyncSandboxClient("local")
        job = UnsupportedBackendTests.JOB
        calls = {
            "submit_job": client.submit_job(JobSpec(command="true")),
            "job_status": client.job_status(job),
            "wait_job": client.wait_job(job, timeout_secs=1),
            "job_logs": client.job_logs(job),
            "cancel_job": client.cancel_job(job),
        }
        for op, call in calls.items():
            with self.subTest(op=op):
                with self.assertRaises(FeatureNotSupportedError) as ctx:
                    await call
                self.assertIn("does not support jobs", str(ctx.exception))
                self.assertEqual(ctx.exception.op, op)

    async def test_async_check_needs_no_backend_sdk(self) -> None:
        """A backend without jobs says so even without its optional SDK."""

        class NotInstalled:

            def __init__(self) -> None:
                raise BackendNotInstalledError("pip install the extra")

        with mock.patch.dict(
            "bespokelabs.sandbox.backends.BACKENDS", {"ray": NotInstalled}
        ):
            client = AsyncSandboxClient("ray")
            with self.assertRaises(FeatureNotSupportedError):
                await client.submit_job(JobSpec(command="true"))

    async def test_async_unsupported_backend_is_named_before_the_spec_is_copied(
        self,
    ) -> None:
        """A spec holding a value deepcopy cannot copy still gets the error."""
        import threading

        spec = JobSpec(
            command="true", backend_options={"lock": threading.Lock()}
        )
        client = AsyncSandboxClient("ray")
        with self.assertRaises(FeatureNotSupportedError):
            await client.submit_job(spec)

    async def test_async_submit_builds_from_a_copy_of_the_spec(self) -> None:
        """A caller changing its spec mid-submit changes no job.

        The submit is held on the backend client's construction while the
        caller changes the spec.
        """
        seen: list[JobSpec] = []
        spec = JobSpec(command="true", env_vars={"A": "1"}, labels={"t": "a"})
        building = asyncio.Event()
        built = asyncio.Event()

        class Runner:

            def submit_job(self, given: JobSpec) -> str:
                seen.append(given)
                return "job-1"

        client = AsyncSandboxClient("gcp_batch")

        async def jobs(op: str) -> Runner:
            building.set()
            await built.wait()
            return Runner()

        with mock.patch.object(client, "_jobs", jobs):
            submit = asyncio.create_task(client.submit_job(spec))
            await building.wait()
            spec.env_vars["A"] = "2"
            spec.labels["t"] = "b"
            built.set()
            self.assertEqual(await submit, "job-1")
        self.assertIsNot(seen[0], spec)
        self.assertEqual(seen[0].env_vars, {"A": "1"})
        self.assertEqual(seen[0].labels, {"t": "a"})


class WaitPolicyTests(unittest.TestCase):

    def _wait(
        self,
        outcomes: list[object],
        *,
        timeout_secs: float | None = None,
    ) -> tuple[object, list[JobStatus]]:
        polled: list[JobStatus] = []
        queue = list(outcomes)

        def status(job_id: str) -> JobStatus:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return JobStatus(job_id=job_id, state=item)

        with mock.patch.object(_jobs.time, "sleep"):
            result = _jobs.wait_sync(
                status,
                "job",
                timeout_secs=timeout_secs,
                poll_secs=1,
                on_poll=polled.append,
            )
        return result, polled

    def test_returns_terminal_status_and_reports_every_poll(self) -> None:
        result, polled = self._wait(
            [JobState.PENDING, JobState.RUNNING, JobState.SUCCEEDED]
        )
        self.assertEqual(result.state, JobState.SUCCEEDED)
        self.assertEqual(
            [p.state for p in polled],
            [JobState.PENDING, JobState.RUNNING, JobState.SUCCEEDED],
        )

    def test_unknown_is_terminal(self) -> None:
        result, _ = self._wait([JobState.RUNNING, JobState.UNKNOWN])
        self.assertEqual(result.state, JobState.UNKNOWN)

    def test_gives_up_after_five_failures_before_the_job_is_seen(self) -> None:
        errors = [SandboxConnectionError(f"502 #{i}") for i in range(5)]
        with self.assertRaises(SandboxConnectionError) as ctx:
            self._wait([*errors, JobState.SUCCEEDED])
        self.assertIn("#4", str(ctx.exception))

    def test_tolerates_four_failures_before_the_job_is_seen(self) -> None:
        errors = [SandboxConnectionError("502")] * 4
        result, _ = self._wait([*errors, JobState.SUCCEEDED])
        self.assertEqual(result.state, JobState.SUCCEEDED)

    def test_rides_out_failures_once_the_job_was_seen(self) -> None:
        errors = [SandboxConnectionError("502")] * 20
        result, _ = self._wait([JobState.RUNNING, *errors, JobState.FAILED])
        self.assertEqual(result.state, JobState.FAILED)

    def test_fatal_errors_raise_at_once(self) -> None:
        with self.assertRaises(SandboxAuthenticationError):
            self._wait(
                [JobState.RUNNING, SandboxAuthenticationError("bad key")]
            )

    def test_backoff_doubles_and_is_capped(self) -> None:
        waiter = _jobs._Waiter("job", None, poll_secs=10)
        waiter.observed(JobStatus(job_id="job", state=JobState.RUNNING))
        delays = [waiter.failed(SandboxConnectionError("x")) for _ in range(8)]
        self.assertEqual(delays[:4], [10, 20, 40, 60])
        self.assertTrue(all(d == 60 for d in delays[4:]))

    def test_deadline(self) -> None:
        with self.assertRaises(SandboxTimeoutError) as ctx:
            self._wait([JobState.RUNNING] * 3, timeout_secs=0)
        self.assertEqual(ctx.exception.context["state"], "running")


class ControlPlaneDefaultsTests(unittest.TestCase):

    def test_job_only_backends_are_not_enabled_by_default(self) -> None:
        from bespokelabs.sandbox.control_plane.service import ControlPlane

        plane = ControlPlane(store=mock.MagicMock())
        self.assertNotIn("gcp_batch", plane._allowed_backends)
        self.assertIn("local", plane._allowed_backends)

    def test_job_only_backends_are_rejected_when_named(self) -> None:
        from bespokelabs.sandbox.control_plane.service import ControlPlane

        with self.assertRaisesRegex(ValueError, "gcp_batch"):
            ControlPlane(
                store=mock.MagicMock(), allowed_backends=["local", "gcp_batch"]
            )


if __name__ == "__main__":
    unittest.main()
