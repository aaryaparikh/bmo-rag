"""Serve each open-source embedding model in vLLM and benchmark it sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bmo_rag.indexing.embeddings import MODEL_SPECS, ModelSpec, resolve_model

CONTAINER_NAME = "bmo-rag-vllm"
DEFAULT_IMAGE = "vllm/vllm-openai:v0.26.0"
VLLM_URL = "http://127.0.0.1:8000"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=check, text=True)


def remove_server() -> None:
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def server_command(
    spec: ModelSpec,
    image: str,
    batch_size: int | None = None,
    gpu_memory_utilization: float = 0.80,
) -> list[str]:
    effective_batch_size = batch_size or spec.recommended_batch_size
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        CONTAINER_NAME,
        "--gpus",
        "all",
        "--ipc=host",
        "--publish",
        "127.0.0.1:8000:8000",
        "--volume",
        "bmo-rag-huggingface-cache:/root/.cache/huggingface",
        "--volume",
        "bmo-rag-vllm-cache:/root/.cache/vllm",
        image,
        "--model",
        spec.model_id,
        "--served-model-name",
        spec.model_id,
        "--runner",
        "pooling",
        "--dtype",
        "half",
        "--max-model-len",
        str(spec.max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max(effective_batch_size, 8)),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    if spec.quantize_4bit:
        command.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    if spec.trust_remote_code:
        command.append("--trust-remote-code")
    if spec.hf_overrides:
        command.extend(["--hf-overrides", spec.hf_overrides])
    if spec.pooler_config:
        command.extend(["--pooler-config", spec.pooler_config])
    return command


def container_running() -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip().casefold() == "true"


def wait_until_ready(spec: ModelSpec, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    next_update = 0.0
    while time.monotonic() < deadline:
        if not container_running():
            print(f"vLLM startup logs for {spec.model_id}:", flush=True)
            run(["docker", "logs", "--tail", "200", CONTAINER_NAME], check=False)
            raise RuntimeError(f"vLLM exited while loading {spec.model_id}")
        try:
            with urlopen(f"{VLLM_URL}/v1/models", timeout=5) as response:
                payload = json.load(response)
            loaded = {item["id"] for item in payload.get("data", [])}
            if spec.model_id in loaded:
                print(f"vLLM is ready with {spec.model_id}", flush=True)
                return
        except (OSError, URLError, ValueError, json.JSONDecodeError):
            pass
        now = time.monotonic()
        if now >= next_update:
            print(
                f"Waiting for {spec.model_id} to download/load; first startup can take a while...",
                flush=True,
            )
            next_update = now + 30
        time.sleep(5)
    run(["docker", "logs", "--tail", "200", CONTAINER_NAME], check=False)
    raise TimeoutError(f"vLLM did not become ready within {timeout_seconds}s for {spec.model_id}")


def benchmark_command(args: argparse.Namespace, spec: ModelSpec) -> list[str]:
    effective_batch_size = args.batch_size or spec.recommended_batch_size
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluation/benchmark_embeddings.py"),
        "--models",
        spec.slug,
        "--base-url",
        f"{VLLM_URL}/v1",
        "--batch-size",
        str(effective_batch_size),
        "--output",
        str(args.output),
        "--append-report",
    ]
    if args.reindex:
        command.append("--reindex")
    if args.details_output_dir:
        command.extend(["--details-output-dir", str(args.details_output_dir)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override VRAM-safe defaults (Qwen 8B: 8, Qwen 4B: 16, others: 32).",
    )
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
        help="Fraction of GPU memory reserved by vLLM (default: 0.80 for an 8 GB Windows GPU).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/benchmarks/embedding_model_comparison/summary.json",
    )
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument(
        "--details-output-dir",
        type=Path,
        default=ROOT / "outputs/benchmarks/embedding_model_comparison/query_details",
        help="Write query-level JSONL audit datasets while evaluating each model.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the Docker and benchmark commands only."
    )
    args = parser.parse_args()

    specs = [resolve_model(model) for model in args.models]
    if args.dry_run:
        for spec in specs:
            print(
                subprocess.list2cmdline(
                    server_command(
                        spec,
                        args.image,
                        args.batch_size,
                        args.gpu_memory_utilization,
                    )
                )
            )
            print(subprocess.list2cmdline(benchmark_command(args, spec)))
        return
    run(["docker", "compose", "up", "-d", "qdrant"])
    try:
        for position, spec in enumerate(specs, start=1):
            print(f"\n[{position}/{len(specs)}] Starting {spec.model_id}", flush=True)
            remove_server()
            run(
                server_command(
                    spec,
                    args.image,
                    args.batch_size,
                    args.gpu_memory_utilization,
                )
            )
            wait_until_ready(spec, args.startup_timeout)
            run(benchmark_command(args, spec))
            print(f"Completed {spec.slug}", flush=True)
    finally:
        remove_server()

    print(f"\nAll requested models completed. Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
