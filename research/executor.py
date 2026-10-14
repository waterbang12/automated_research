"""Runs code in a locked-down Docker container. The host never executes generated code.

Container restrictions: no network, read-only root fs, all capabilities dropped,
no-new-privileges, unprivileged uid, memory/CPU/pid limits, wall-clock timeout.
Only three host paths are mounted: the experiment's code (read-only), the public
data (read-only) and an empty output directory.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from .config import SandboxConfig


class ExecutorUnavailable(RuntimeError):
    pass


@dataclass
class ExecResult:
    exit_code: int | None
    timed_out: bool
    duration_s: float
    stdout: str
    stderr: str
    command: list[str]
    image_id: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


async def _run(*args: str, timeout: float = 60) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(proc.communicate(), timeout)
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


def _tail(s: str, n: int) -> str:
    return s if len(s) <= n else f"...[{len(s) - n} chars truncated]...\n" + s[-n:]


class DockerExecutor:
    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        self.image_id = ""

    async def preflight(self) -> str:
        try:
            code, _, err = await _run("docker", "info", "--format", "{{.ServerVersion}}", timeout=30)
        except (FileNotFoundError, asyncio.TimeoutError) as e:
            raise ExecutorUnavailable(f"docker not available: {e}") from e
        if code != 0:
            raise ExecutorUnavailable(f"docker daemon not reachable: {err.strip()}")
        code, out, err = await _run("docker", "image", "inspect", "--format", "{{.Id}}", self.cfg.image)
        if code != 0:
            raise ExecutorUnavailable(
                f"sandbox image {self.cfg.image!r} missing; build it with `python -m research build-sandbox`")
        self.image_id = out.strip()
        return self.image_id

    def command(self, name: str, mounts: list[tuple[Path, str, bool]], env: dict[str, str],
                argv: list[str]) -> list[str]:
        c = self.cfg
        cmd = ["docker", "run", "--rm", "--name", name,
               "--network", c.network,
               "--memory", c.memory, "--memory-swap", c.memory,
               "--cpus", str(c.cpus), "--pids-limit", str(c.pids),
               "--read-only", "--tmpfs", "/tmp:rw,size=256m",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--user", "65534:65534", "-w", "/tmp"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        for src, dst, ro in mounts:
            cmd += ["--mount", f"type=bind,source={Path(src).resolve()},target={dst}" + (",readonly" if ro else "")]
        return cmd + [c.image, *argv]

    async def run(self, *, label: str, mounts: list[tuple[Path, str, bool]], env: dict[str, str],
                  argv: list[str], timeout_s: float | None = None) -> ExecResult:
        if not self.image_id:
            await self.preflight()
        name = f"ar-{label}-{secrets.token_hex(3)}"
        cmd = self.command(name, mounts, env, argv)
        timeout = timeout_s or self.cfg.timeout_s
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        timed_out = False
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await self._kill(name)
            out, err = await proc.communicate()
        except asyncio.CancelledError:
            await self._kill(name)
            raise
        n = self.cfg.max_output_bytes
        return ExecResult(
            exit_code=proc.returncode, timed_out=timed_out, duration_s=round(time.monotonic() - t0, 2),
            stdout=_tail(out.decode(errors="replace"), n), stderr=_tail(err.decode(errors="replace"), n),
            command=cmd, image_id=self.image_id)

    async def _kill(self, name: str) -> None:
        try:
            await _run("docker", "kill", name, timeout=30)
        except Exception:
            pass
