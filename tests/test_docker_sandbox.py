"""Real Docker tests of the sandbox restrictions. Skipped when Docker or the image is unavailable."""
import asyncio

import pytest

from research.config import SandboxConfig
from research.executor import DockerExecutor, ExecutorUnavailable


@pytest.fixture(scope="module")
def executor():
    ex = DockerExecutor(SandboxConfig(timeout_s=60))
    try:
        asyncio.run(ex.preflight())
    except ExecutorUnavailable as e:
        pytest.skip(str(e))
    return ex


def _run(ex, tmp_path, code, timeout=None):
    (tmp_path / "code").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "code" / "x.py").write_text(code)
    (tmp_path / "data" / "d.txt").write_text("data")
    return asyncio.run(ex.run(label="test", argv=["python", "/code/x.py"], env={"SEED": "7"}, timeout_s=timeout,
                              mounts=[(tmp_path / "code", "/code", True), (tmp_path / "data", "/data", True),
                                      (tmp_path / "out", "/out", False)]))


def test_sandbox_restrictions(executor, tmp_path):
    code = r'''
import os, socket
def attempt(f):
    try:
        f(); return "ALLOWED"
    except Exception as e:
        return "denied"
print("net", attempt(lambda: socket.create_connection(("1.1.1.1", 53), timeout=3)))
print("data_write", attempt(lambda: open("/data/d.txt", "w")))
print("code_write", attempt(lambda: open("/code/y.py", "w")))
print("root_write", attempt(lambda: open("/etc/x", "w")))
print("uid", os.getuid(), "seed", os.environ["SEED"])
open("/out/r.txt", "w").write("ok")
'''
    r = _run(executor, tmp_path, code)
    assert r.ok, r.stderr
    for k in ("net", "data_write", "code_write", "root_write"):
        assert f"{k} denied" in r.stdout
    assert "uid 65534 seed 7" in r.stdout
    assert (tmp_path / "out" / "r.txt").read_text() == "ok"
    assert (tmp_path / "data" / "d.txt").read_text() == "data"


def test_timeout_kills_container(executor, tmp_path):
    r = _run(executor, tmp_path, "import time\nprint('start', flush=True)\ntime.sleep(60)\n", timeout=5)
    assert r.timed_out and not r.ok
    assert r.duration_s < 30


def test_memory_limit(executor, tmp_path):
    r = _run(executor, tmp_path, "x = bytearray(3 * 1024**3)\nprint('allocated')\n")
    assert not r.ok and "allocated" not in r.stdout
