"""四维注意力监督检测器（对应《组件设计/1冲突检测器.md》）。

结构：
- 维度内融合：每个维度（医疗/意愿/QoL/情境）的特征经 Linear+MLP 融合为一个 d_model 向量；
- 跨维度注意力：4 个维度向量作为 token 过 Multi-Head Self-Attention（捕捉维度间伦理张力，
  如 MUI高↔VPI低 的冲突信号被放大）；
- 输出头（多任务）：ERS 回归（sigmoid）+ 四原则多标签（sigmoid）。

训练数据为第三方标注（dataset.py），特征 = 四盒规则提取的 16 维向量。
依赖 torch>=2.0（仅训练/推理此模型需要；rule/LLM 检测器不依赖 torch）。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .rules_features import DIM_GROUPS


class FourDimAttentionDetector(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # 维度内：特征 → 维度表示
        self.dim_lin = nn.ModuleDict({
            dim: nn.Linear(len(fs), d_model) for dim, fs in DIM_GROUPS.items()
        })
        # 维度内融合 MLP（捕捉同维度特征交互）
        self.intra_mlp = nn.ModuleDict({
            dim: nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(d_model, d_model))
            for dim in DIM_GROUPS
        })
        # 跨维度注意力（4 个维度 token，捕捉伦理张力）
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        # 多任务头
        self.head_ers = nn.Linear(d_model, 1)
        self.head_principle = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, 16) 按 DIM_GROUPS 顺序拼接的四盒特征。"""
        vecs = []
        start = 0
        for dim, fs in DIM_GROUPS.items():
            seg = x[:, start:start + len(fs)]
            start += len(fs)
            v = self.intra_mlp[dim](self.dim_lin[dim](seg))  # (B, d_model)
            vecs.append(v)
        tokens = torch.stack(vecs, dim=1)                    # (B, 4, d_model)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        pooled = tokens.mean(dim=1)                          # (B, d_model)
        out = self.ffn(pooled)
        return {
            "ers": torch.sigmoid(self.head_ers(out).squeeze(-1)),          # (B,)
            "principle": torch.sigmoid(self.head_principle(out)),           # (B, 4)
        }


def loss_fn(pred: Dict[str, torch.Tensor], y_principle: torch.Tensor, m_principle: torch.Tensor,
            y_ers: torch.Tensor, m_ers: torch.Tensor,
            w_principle: float = 1.0, w_ers: float = 1.0) -> torch.Tensor:
    """多任务损失：原则多标签(BCE，仅掩码样本) + ERS 回归(MSE，仅掩码样本)。

    y 均为已掩码处理（无标签处填 0），mask 指示有效样本。
    """
    loss = torch.tensor(0.0, device=pred["principle"].device)
    if m_principle.any():
        p = pred["principle"][m_principle]
        y = y_principle[m_principle]
        loss = loss + w_principle * nn.functional.binary_cross_entropy(p, y)
    if m_ers.any():
        e = pred["ers"][m_ers]
        ye = y_ers[m_ers]
        loss = loss + w_ers * nn.functional.mse_loss(e, ye)
    return loss


def principle_accuracy(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """宏平均逐原则准确率（仅掩码样本）。"""
    if not mask.any():
        return 0.0
    p = (pred[mask] > 0.5).float()
    y = y[mask]
    return float((p == y).float().mean())


def ers_mae(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float((pred[mask] - y[mask]).abs().mean())
