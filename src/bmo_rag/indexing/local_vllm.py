"""Start and reuse a local vLLM embedding server for ad-hoc retrieval."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from bmo_rag.indexing.embeddings import ModelSpec

CONTAINER_NAME = "bmo-rag-vllm"
RERANKER_CONTAINER_NAME = "bmo-rag-reranker"
DEFAULT_IMAGE = "vllm/vllm-openai:v0.26.0"


class LocalVllmError(RuntimeError):
    """Raised when the local Docker-backed embedding service cannot start."""


def loaded_models(base_url: str, *, timeout: float = 3.0) -> set[str]:
    """Return model IDs advertised by an OpenAI-compatible endpoint."""
    models_url = f"{base_url.rstrip('/').removesuffix('/v1')}/v1/models"
    try:
        with urlopen(models_url, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return set()
    return {item["id"] for item in payload.get("data", []) if item.get("id")}


def _run(command: list[str], *, project_root: Path, capture: bool = False) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=capture,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise LocalVllmError(f"Docker command failed: {detail.strip()}") from exc
    return result.stdout.strip() if capture else ""


def ensure_local_embedding_services(
    spec: ModelSpec,
    *,
    project_root: Path,
    base_url: str = "http://127.0.0.1:8000/v1",
    image: str = DEFAULT_IMAGE,
    startup_timeout: int = 3600,
    gpu_memory_utilization: float = 0.80,
    progress: Callable[[str], None] | None = None,
) -> bool:
    """Ensure Qdrant and the selected model are running; return whether vLLM was started."""
    if spec.model_id in loaded_models(base_url):
        return False

    notify = progress or (lambda _message: None)
    _run(["docker", "compose", "up", "-d", "qdrant"], project_root=project_root)
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    command = [
        "docker", "run", "--detach", "--name", CONTAINER_NAME,
        "--gpus", "all", "--ipc=host", "--publish", "127.0.0.1:8000:8000",
        "--volume", "bmo-rag-huggingface-cache:/root/.cache/huggingface",
        "--volume", "bmo-rag-vllm-cache:/root/.cache/vllm", image,
        "--model", spec.model_id, "--served-model-name", spec.model_id,
        "--runner", "pooling", "--dtype", "half",
        "--max-model-len", str(spec.max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-num-seqs", str(max(spec.recommended_batch_size, 8)),
        "--host", "0.0.0.0", "--port", "8000",
    ]
    if spec.quantize_4bit:
        command.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    if spec.trust_remote_code:
        command.append("--trust-remote-code")
    if spec.hf_overrides:
        command.extend(["--hf-overrides", spec.hf_overrides])
    if spec.pooler_config:
        command.extend(["--pooler-config", spec.pooler_config])
    _run(command, project_root=project_root)

    notify(f"Loading {spec.model_id} in vLLM (the first startup can take a few minutes)...")
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if spec.model_id in loaded_models(base_url, timeout=5):
            notify(f"vLLM is ready with {spec.model_id}.")
            return True
        time.sleep(3)
    raise LocalVllmError(
        f"vLLM did not become ready within {startup_timeout} seconds. "
        f"Inspect it with: docker logs {CONTAINER_NAME}"
    )


def ensure_local_reranker(
    *,
    model: str,
    project_root: Path,
    base_url: str = "http://127.0.0.1:8001",
    image: str = DEFAULT_IMAGE,
    startup_timeout: int = 3600,
    gpu_memory_utilization: float = 0.35,
    progress: Callable[[str], None] | None = None,
) -> bool:
    """Start or reuse a local vLLM cross-encoder reranker on port 8001."""
    if model in loaded_models(base_url):
        return False
    notify = progress or (lambda _message: None)
    subprocess.run(
        ["docker", "rm", "-f", RERANKER_CONTAINER_NAME],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    _run(
        [
            "docker", "run", "--detach", "--name", RERANKER_CONTAINER_NAME,
            "--gpus", "all", "--ipc=host", "--publish", "127.0.0.1:8001:8000",
            "--env", "HF_HUB_DISABLE_XET=1",
            "--volume", "bmo-rag-huggingface-cache:/root/.cache/huggingface",
            "--volume", "bmo-rag-vllm-cache:/root/.cache/vllm", image,
            "--model", model, "--served-model-name", model,
            "--runner", "pooling", "--dtype", "half", "--max-model-len", "2048",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-num-seqs", "16", "--host", "0.0.0.0", "--port", "8000",
        ],
        project_root=project_root,
    )
    notify(f"Loading reranker {model} in vLLM...")
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if model in loaded_models(base_url, timeout=5):
            notify(f"vLLM reranker is ready with {model}.")
            return True
        time.sleep(3)
    raise LocalVllmError(
        f"Reranker did not become ready within {startup_timeout} seconds. "
        f"Inspect it with: docker logs {RERANKER_CONTAINER_NAME}"
    )
