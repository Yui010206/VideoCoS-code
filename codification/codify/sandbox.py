"""OS-enforced sandbox for generated Blender code: a Docker container per call.

The container has no network, a read-only root file system, no added capabilities, the
caller's uid/gid, CPU/memory/PID limits and a host-side timeout. It sees only the
Blender install (read-only), this codify package (read-only), one evidence version
(read-only) and one revision directory (read-write). No credentials are passed; the only
environment variables are HOME, PYTHONDONTWRITEBYTECODE and, in a room run (settings fast_room),
CODIFY_ROOM=1, which codify.arrange reads. If Docker or the image is unavailable, execution is refused.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

CODE = Path(__file__).resolve().parent
IMAGE = "videocos-blender-runtime:1"
DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
    libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libgl1 libegl1 libglx-mesa0 \\
    && rm -rf /var/lib/apt/lists/*
"""


class SandboxUnavailable(RuntimeError):
    pass


class Sandbox:
    def __init__(self, blender_dir, cpus=8, memory="8g", timeout=900, max_output_mb=512, image=IMAGE, room=False):
        self.blender_dir = Path(blender_dir).resolve()
        self.room = room                      # a room run (fast_room): Blender gets CODIFY_ROOM=1
        self.cpus, self.memory, self.timeout = cpus, memory, timeout
        self.max_output_bytes, self.image = max_output_mb * 2 ** 20, image

    def check(self):
        """Raise SandboxUnavailable unless Docker, the image and Blender are usable."""
        if not shutil.which("docker"):
            raise SandboxUnavailable("docker CLI not found; generated code will not be executed without isolation")
        probe = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True, text=True)
        if probe.returncode:
            raise SandboxUnavailable(f"sandbox image {self.image} missing; build it with Sandbox.build_image()")
        if not (self.blender_dir / "blender").is_file():
            raise SandboxUnavailable(f"Blender not found in {self.blender_dir}")

    def build_image(self):
        """One-time setup (needs network for apt); execution itself never has network."""
        subprocess.run(["docker", "build", "-q", "-t", self.image, "-"], input=DOCKERFILE, text=True, check=True)

    def _container(self, name, workdir, evidence):
        """docker run arguments: no network, read-only root, the caller's uid, limits; workdir at /out (read-write),
        the evidence at /evidence (read-only)."""
        command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
                   "--tmpfs", "/tmp:rw,size=2g", "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--pids-limit", "2048", "--cpus", str(self.cpus),
                   "--memory", self.memory, "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-v", f"{self.blender_dir}:/opt/blender:ro", "-v", f"{CODE}:/harness/codify:ro",
                   "-v", f"{workdir}:/out:rw"]
        if self.room:
            command += ["-e", "CODIFY_ROOM=1"]
        if evidence is not None:
            command += ["-v", f"{Path(evidence).resolve()}:/evidence:ro"]
        return command

    def python(self, script, workdir, evidence=None, timeout=300):
        """Run workdir/script with Blender's bundled Python (numpy) in the same container: for the agent's own
        measurements on the evidence, where the coding CLI's shell is unavailable. Returns as blender() does."""
        self.check()
        workdir = Path(workdir).resolve()
        interpreter = next(self.blender_dir.glob("*/python/bin/python3*"))
        name = "codify-py-" + uuid.uuid4().hex[:12]
        command = self._container(name, workdir, evidence) + [
            self.image, "/opt/blender/" + str(interpreter.relative_to(self.blender_dir)), f"/out/{script}"]
        return self._run(name, command, workdir, "python", timeout)

    def blender(self, op_args, workdir, evidence=None, blend=None, timeout=None):
        """Run blender_ops.py inside the container; workdir is mounted at /out (read-write)."""
        self.check()
        workdir = Path(workdir).resolve()
        name = "codify-" + uuid.uuid4().hex[:12]
        command = self._container(name, workdir, evidence)
        # Blender sizes its thread pool from the host CPU count unless told otherwise.
        command += [self.image, "/opt/blender/blender", "--background", "--factory-startup", "--threads", str(self.cpus)]
        if blend is not None:
            command.append(f"/out/{blend}")
        command += ["--python-exit-code", "4", "--python", "/harness/codify/blender_ops.py", "--", *op_args]
        return self._run(name, command, workdir, op_args[0], timeout)

    def _run(self, name, command, workdir, label, timeout):
        started = time.time()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout or self.timeout)
            status, code = ("ok" if result.returncode == 0 else "error"), result.returncode
            stdout, stderr = result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True)
            status, code = "timeout", None
            stdout, stderr = (exc.stdout or b"").decode(errors="replace"), (exc.stderr or b"").decode(errors="replace")
        size = sum(p.stat().st_size for p in workdir.rglob("*") if p.is_file())
        if size > self.max_output_bytes:
            status = "error"
            stderr += f"\nOutput size {size / 2**20:.1f} MB exceeds the {self.max_output_bytes / 2**20:.0f} MB cap"
        (workdir / "logs").mkdir(exist_ok=True)
        log = workdir / "logs" / f"{label}_{int(started)}.log"
        log.write_text(stdout + "\n--- stderr ---\n" + stderr)
        return {"status": status, "returncode": code, "seconds": round(time.time() - started, 2), "log": str(log),
                "stderr_tail": stderr[-1500:], "stdout_tail": stdout[-20000:], "output_bytes": size}
