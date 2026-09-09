"""05_train_detector：训练四维注意力监督检测器（第三方标注）。

数据：PrinciplismQA（四原则标注）+ VITAL（value→原则映射）+ MedEthicEval（ERS 弱监督）。
用法：
  python scripts/05_train_detector.py --config configs/config.yaml \
      --data-dir <数据集根目录> --sources principlismqa,vital,medethiceval \
      --epochs 30 --lr 1e-3 --batch 64 --out runs/detector.pt
训练后把 runs/detector.pt 填入 config.yaml 的 detection.checkpoint（或环境变量
ETHICALGUARD_DETECTOR_CHECKPOINT），02_run_mane 即自动使用监督检测器。

依赖：torch>=2.0（pip install torch）。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ethicalguard.config import Config
from ethicalguard.utils import set_seed, setup_logging

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data-dir", default=None, help="数据集根目录（默认 config.datasets.data_dir）")
    ap.add_argument("--dataset-paths", default=None, help="逐数据集路径覆盖 name=/path（逗号分隔）")
    ap.add_argument("--sources", default="principlismqa,vital,medethiceval")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--w-ers", type=float, default=1.0,
                    help="ERS 头损失权重（默认 1.0；ERS 样本少（436）时加大可强化风险头，"
                         "如 --w-ers 3，提高 ERS 区分度）")
    ap.add_argument("--out", default="runs/detector.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from ethicalguard.detection.model import FourDimAttentionDetector, ers_mae, loss_fn, principle_accuracy
    except ImportError as e:  # noqa: F841
        print("[错误] 需要 torch>=2.0：pip install torch")
        sys.exit(1)

    set_seed(args.seed)
    cfg = Config.load(args.config)
    data_dir = args.data_dir or cfg.datasets.data_dir
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    # 逐数据集路径覆盖（与 01_prepare 同语义：dataset_paths 作用于单个数据集根目录）
    dataset_paths = dict(cfg.datasets.paths or {})
    if args.dataset_paths:
        for item in args.dataset_paths.split(","):
            item = item.strip()
            if item and "=" in item:
                k, v = item.split("=", 1)
                dataset_paths[k.strip()] = v.strip()
    # 注意：dataset.py 目前按 data_dir/<Folder> 组织；如需逐数据集覆盖，可在此展开（见 build_dataset）
    # 简化：若提供了 dataset_paths，则把每个数据集改为独立调用（此处仅打印提示）
    if dataset_paths:
        logger.warning("05 训练脚本当前按统一 data_dir 组织数据集；逐数据集路径覆盖请手动调整 data_dir")

    from ethicalguard.detection.dataset import DetectorDataset, build_dataset, fit_ers_temperature
    ds = build_dataset(data_dir, sources=sources)
    if len(ds) == 0:
        print("[错误] 未构建到任何监督样本，请检查 --data-dir 与数据集完整性。")
        sys.exit(1)

    X = ds.features().astype(np.float32)
    yp = ds.principle_labels()
    mp = ds.principle_mask()
    ye = ds.ers_labels()
    me = ds.ers_mask()
    logger.info("监督样本 %d（原则标签 %d，ERS 标签 %d）", len(ds),
                int(mp.sum()), int(me.sum()))

    # 划分训练/评估
    n = len(ds)
    idx = np.random.permutation(n)
    n_eval = max(1, int(n * args.eval_frac))
    eval_idx, train_idx = idx[:n_eval], idx[n_eval:]

    def to_tensor(a):
        return torch.from_numpy(a) if a is not None else None

    Xt = torch.from_numpy(X[train_idx])
    yp_t = to_tensor(yp[train_idx]) if yp is not None else torch.zeros(len(train_idx), 4)
    mp_t = torch.from_numpy(mp[train_idx]).bool()
    ye_t = to_tensor(ye[train_idx]) if ye is not None else torch.zeros(len(train_idx))
    me_t = torch.from_numpy(me[train_idx]).bool()

    Xe = torch.from_numpy(X[eval_idx])
    yp_e = to_tensor(yp[eval_idx]) if yp is not None else torch.zeros(len(eval_idx), 4)
    mp_e = torch.from_numpy(mp[eval_idx]).bool()
    ye_e = to_tensor(ye[eval_idx]) if ye is not None else torch.zeros(len(eval_idx))
    me_e = torch.from_numpy(me[eval_idx]).bool()

    model = FourDimAttentionDetector(d_model=args.d_model, n_heads=args.n_heads)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(perm), args.batch):
            bidx = perm[i:i + args.batch]
            xb = Xt[bidx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yp_t[bidx], mp_t[bidx], ye_t[bidx], me_t[bidx], w_ers=args.w_ers)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(bidx)
        train_loss = total / max(1, len(train_idx))

        # 评估
        model.eval()
        with torch.no_grad():
            pred_e = model(Xe)
            acc = principle_accuracy(pred_e["principle"], yp_e, mp_e)
            mae = ers_mae(pred_e["ers"], ye_e, me_e)
        logger.info("epoch %d/%d train_loss=%.4f eval: principle_acc=%.3f ers_mae=%.3f",
                    epoch, args.epochs, train_loss, acc, mae)

    # ERS 温度校准（P-7.1）：在 eval 集拟合温度 t，推理时 σ(t·logit) 拉开分布
    ers_temperature = 1.0
    with torch.no_grad():
        pred_e = model(Xe)
        p_e = pred_e["ers"].cpu().numpy().clip(1e-6, 1 - 1e-6)
        logit_e = np.log(p_e / (1.0 - p_e))
        mask_e = me_e.numpy()
        if mask_e.sum() >= 5:
            ers_temperature = fit_ers_temperature(logit_e[mask_e], ye_e[mask_e].numpy())
    logger.info("ERS 温度校准: t=%.3f（eval ERS 样本 %d）", ers_temperature, int(mask_e.sum()))

    # 保存
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "model_args": {"d_model": args.d_model, "n_heads": args.n_heads},
        "state_dict": model.state_dict(),
        "sources": list(sources),
        "n_samples": len(ds),
        "ers_temperature": ers_temperature,
    }, args.out)
    print(f"已保存检测器权重 -> {args.out}")
    print("下一步：把该路径填入 config.yaml 的 detection.checkpoint（或设置 "
          "ETHICALGUARD_DETECTOR_CHECKPOINT），02_run_mane 即自动使用监督检测器。")


if __name__ == "__main__":
    main()
