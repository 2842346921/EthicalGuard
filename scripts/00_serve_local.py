"""00_serve_local：按统一配置启动本地模型服务（vLLM）。

用法：
  python scripts/00_serve_local.py --config configs/config.yaml [--port 8000] [--dry-run]

说明：从配置读取 llm.local.model_path / llm.local.model；仅打印命令时用 --dry-run。
若未填 model_path，则只打印提示，由你手动起服务（Ollama 等）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethicalguard.config import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    # Qwen3 原生 max_position_embeddings=40960，按它预留 KV 缓存需要 5.6G+，
    # 24G 卡权重(15.3G)剩余放不下；实际场景只有几千 token，16384 足够
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    local = cfg.llm.local
    if not local.model_path:
        print("[提示] config.yaml 的 llm.local.model_path 为空，无法自动启动 vLLM。")
        print("       请填写模型权重路径，或手动起服务后设置 ETHICALGUARD_LOCAL_BASE_URL。")
        sys.exit(1)

    if shutil.which("vllm") is None:
        print("[错误] 未找到 vllm 命令（pip install vllm 或使用 ollama serve）。")
        sys.exit(1)

    cmd = [
        "vllm", "serve", local.model_path,
        "--served-model-name", local.model,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
    ]
    print("启动命令：", " ".join(cmd))
    if not args.dry_run:
        print("正在启动本地模型服务（Ctrl+C 停止）...")
        subprocess.run(cmd)  # noqa: S603


if __name__ == "__main__":
    main()
