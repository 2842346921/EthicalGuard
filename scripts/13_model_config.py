"""E5 辅助：生成指定模型的临时配置 + 启动命令（跨模型验证，审稿人 A4/MC4）。

用法（服务器）：
  1. 用本脚本生成某模型的配置副本：
     python scripts/13_model_config.py --model llama31 --out configs/config_llama31.yaml
     python scripts/13_model_config.py --model qwen14  --out configs/config_qwen14.yaml
     python scripts/13_model_config.py --model mistral --out configs/config_mistral.yaml
  2. 在另一端口起该模型的 vLLM：
     python scripts/00_serve_local.py --config configs/config_llama31.yaml --port 8011 --dry-run
     （确认命令后去掉 --dry-run 后台起）
  3. 用该配置跑实验（02/03/06/08/10/11）→ 同一批场景跨模型出数。

可用 --model 键（对应 /mnt/model 下的实际目录）：
  qwen3-8b   : /mnt/model/Qwen3-8B-Instruct  （默认，端口 8003）
  qwen14     : /mnt/model/Qwen3-14B
  llama31    : /mnt/model/Meta-Llama-3.1-8B-Instruct
  llama31b   : /mnt/model/Meta-Llama-3.1-8B   （base 版，无 instruct，谨慎用）
  mistral    : /mnt/model/Mistral-7B-Instruct-v0.3
  qwen25-7b  : /mnt/model/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ethicalguard.config import Config

MODELS = {
    "qwen3-8b":  ("/mnt/model/Qwen3-8B-Instruct", "Qwen3-8B-Instruct", 8003),
    "qwen14":    ("/mnt/model/Qwen3-14B", "Qwen3-14B", 8011),
    "llama31":   ("/mnt/model/Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-8B-Instruct", 8012),
    "mistral":   ("/mnt/model/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3", 8013),
    "qwen25-7b": ("/mnt/model/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", 8014),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS), help="模型键（见 docstring）")
    ap.add_argument("--base", default="configs/config.yaml", help="基础配置（默认 config.yaml）")
    ap.add_argument("--out", default=None, help="输出配置路径（默认 configs/config_<model>.yaml）")
    args = ap.parse_args()

    path, served, port = MODELS[args.model]
    out = args.out or f"configs/config_{args.model}.yaml"
    cfg = Config.load(args.base)
    cfg.llm.local.model_path = path
    cfg.llm.local.model = served
    cfg.llm.local.base_url = f"http://localhost:{port}/v1"

    # 基于基础配置文本做最小替换（保留其它配置不变），避免序列化整棵配置树
    with open(args.base, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 定位 local: 段起始，只改该段内（local 是 llm 下第二个子段；用缩进判断）
    out_lines = []
    in_local = False
    replaced = {"base_url": False, "model": False, "model_path": False}
    for ln in lines:
        stripped = ln.strip()
        indent = len(ln) - len(ln.lstrip())
        if stripped == "local:" and indent == 2:
            in_local = True
            out_lines.append(ln)
            continue
        if in_local and indent <= 2 and stripped:
            in_local = False  # 离开 local 段
        if in_local:
            if stripped.startswith("base_url:") and not replaced["base_url"]:
                out_lines.append(f"{' ' * indent}base_url: {cfg.llm.local.base_url}\n")
                replaced["base_url"] = True
                continue
            if stripped.startswith("model:") and not replaced["model"]:
                out_lines.append(f"{' ' * indent}model: {served}\n")
                replaced["model"] = True
                continue
            if stripped.startswith("model_path:") and not replaced["model_path"]:
                out_lines.append(f"{' ' * indent}model_path: {path}\n")
                replaced["model_path"] = True
                continue
        out_lines.append(ln)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    if not all(replaced.values()):
        print("[警告] 部分字段未替换:", {k: v for k, v in replaced.items() if not v})
    print(f"[已生成] {out}  model={served}\n  path={path}\n  base_url={cfg.llm.local.base_url}")
    print(f"[启动]  python scripts/00_serve_local.py --config {out} --port {port} --dry-run")
    # 校验
    cfg2 = Config.load(out)
    print(f"[校验]  Config.load({out}) → mode={cfg2.llm.mode} model={cfg2.llm.local.model} "
          f"path={cfg2.llm.local.model_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
