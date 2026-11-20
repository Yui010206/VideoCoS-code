"""The perception lanes' Qwen server, in one of two modes.

Per lane: one vLLM OpenAI-compatible server per lane and GPU, loaded once and kept asleep (level 1: the weights in CPU
memory, about 86 GB of RAM per lane; the GPU free for DA3, SAM 3 and SAM 3D), woken for the steps that ask the model:
the vocabulary, the proposal names and the object layer's checks.

Shared: one server on a GPU of its own, always awake, used by every lane (perception.py qwen-server, then perception.py
run --qwen-url): no loads, no sleeping, one copy of the model; CODIFY_QWEN_SHARED=1 tells answers() not to wake or
sleep it.

perception.py starts it (start), sets CODIFY_QWEN_URL and CODIFY_QWEN_GPU for its steps, and stops it at the end; the
workers (models.run_vlm_vocabulary, models.run_vlm_crop_names, vlm_check_worker) call answers(), which wakes the server,
sends every request at once (vLLM batches them) and puts it back to sleep. Standard library only: it runs in the
harness environment (the lane) and in the vLLM environment (the workers)."""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL_ENV, GPU_ENV, SHARED_ENV = "CODIFY_QWEN_URL", "CODIFY_QWEN_GPU", "CODIFY_QWEN_SHARED"
SHARED_PORT = 8701
AWAKE_GB = 72          # GPU memory the awake server needs (weights 55 GB, 6 GiB KV cache, vLLM's buffers)
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_gb(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0) / 1024


def _request(url, path, payload=None, timeout=900, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body.strip() else {}


def start(python, weights, gpu, log_path, max_images=3, kv_cache_gb=6, max_model_len=16384, timeout=1800, port=None,
          max_num_seqs=64):
    """Start the server on gpu and wait until it answers; returns (process, url). The caller puts it to sleep."""
    port = port or _free_port()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_SERVER_DEV_MODE="1", VLLM_USE_FLASHINFER_SAMPLER="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TQDM_DISABLE="1",
               PATH=str(Path(python).parent) + os.pathsep + os.environ.get("PATH", ""))
    command = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(weights), "--served-model-name", "qwen",
               "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16", "--max-model-len", str(max_model_len),
               "--enforce-eager", "--seed", "0", "--kv-cache-memory-bytes", str(int(kv_cache_gb * 2 ** 30)),
               "--limit-mm-per-prompt", json.dumps({"image": max_images}), "--enable-sleep-mode",
               "--max-num-seqs", str(max_num_seqs), "--disable-uvicorn-access-log"]
    log = open(log_path, "a")
    log.write(f"\n=== {time.strftime('%F %T')} gpu {gpu} port {port}\n")
    log.flush()
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the Qwen server exited with {process.returncode}; see {log_path}")
        try:
            urllib.request.urlopen(url + "/health", timeout=5).read()
            return process, url
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(5)
    process.terminate()
    raise RuntimeError(f"the Qwen server did not come up in {timeout} s; see {log_path}")


def healthy(url):
    try:
        urllib.request.urlopen(url + "/health", timeout=10).read()
        return True
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def serve(python, weights, gpu, log_path, port=SHARED_PORT, kv_cache_gb=12):
    """The shared server: start it on gpu at port, always awake, and start it again whenever it stops."""
    while True:
        try:
            process, url = start(python, weights, gpu, log_path, kv_cache_gb=kv_cache_gb, port=port, max_num_seqs=128)
        except RuntimeError as error:
            print(time.strftime("%F %T"), f"the shared Qwen server did not start ({error}); again in 2 minutes", flush=True)
            time.sleep(120)
            continue
        print(time.strftime("%F %T"), f"shared Qwen server up at {url} on gpu {gpu}", flush=True)
        process.wait()
        print(time.strftime("%F %T"), f"the shared Qwen server exited ({process.returncode}); starting it again",
              flush=True)
        time.sleep(30)


def sleep(url, level=1):
    _request(url, f"/sleep?level={level}", method="POST")


def is_sleeping(url):
    return bool(_request(url, "/is_sleeping").get("is_sleeping"))


def wake(url, gpu=None, wait=7200):
    """Wake the server once its GPU has room for it (another job may have taken the memory meanwhile). A wake that
    cannot get the memory raises an out-of-memory error, which the perception step retries."""
    if not is_sleeping(url):
        return
    deadline = time.time() + wait
    while gpu is not None and free_gb(gpu) < AWAKE_GB - 2 and time.time() < deadline:
        time.sleep(30)
    try:
        _request(url, "/wake_up", method="POST")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"CUDA out of memory: the Qwen server could not wake ({error})") from error


def _data_url(path):
    return f"data:{MIME.get(Path(path).suffix.lower(), 'image/jpeg')};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def answers(requests, max_tokens, repetition_penalty=1.0, url=None, gpu=None, workers=64):
    """Greedy answers to [(image paths, prompt text)], in order: wakes the lane's server, sends every request at
    once and puts the server back to sleep. The chat template is the model's own, thinking off."""
    url = url or os.environ[URL_ENV]
    gpu = gpu if gpu is not None else os.environ.get(GPU_ENV)
    shared = os.environ.get(SHARED_ENV) == "1"
    timing = {"wake_started": time.time()}
    if shared:                      # always awake; wait while it is (re)starting
        deadline = time.time() + 3600
        while not healthy(url):
            if time.time() > deadline:
                raise RuntimeError(f"the shared Qwen server at {url} has not answered for an hour")
            time.sleep(30)
    else:
        wake(url, gpu)
    timing["woken"] = time.time()

    def one(request):
        paths, text = request
        content = [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in paths]
        content.append({"type": "text", "text": text})
        body = {"model": "qwen", "messages": [{"role": "user", "content": content}], "temperature": 0.0,
                "max_tokens": max_tokens, "repetition_penalty": repetition_penalty, "seed": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        return _request(url, "/v1/chat/completions", body)["choices"][0]["message"]["content"] or ""
    try:
        with ThreadPoolExecutor(max(1, min(workers, len(requests)))) as pool:
            texts = list(pool.map(one, requests))
    finally:
        timing["answered"] = time.time()
        if not shared:
            sleep(url)
    return texts, {"engine": "vllm server (shared)" if shared else "vllm server (sleep mode)", "wake_seconds": round(timing["woken"] - timing["wake_started"], 2),
                   "generate_seconds": round(timing["answered"] - timing["woken"], 2),
                   "sleep_seconds": round(time.time() - timing["answered"], 2)}


if __name__ == "__main__":       # a quick check: python qwen.py <python> <weights> <gpu> <log>
    proc, url = start(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print("up", url, "free GB awake:", round(free_gb(sys.argv[3]), 1), flush=True)
    sleep(url)
    print("asleep, free GB:", round(free_gb(sys.argv[3]), 1), flush=True)
    proc.terminate()
