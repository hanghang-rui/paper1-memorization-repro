# -*- coding: utf-8 -*-
"""
step2_sweep.py —— 第 2 步：在随机数据上测模型容量，复现「数据量—记忆量」曲线

对应论文 3.2 节，参数按论文设置，规模按本机能力等比缩小：

    论文：V=2048, S=64, 1e6 步, batch 2048, 100K–20M 参数, 数据量 6.5 万–838 万
    本机：V=2048, S=64 保持不变（保证每条数据 693 bits 的比特尺度一致）
          模型缩到约 66 万参数 → 容量约 2.3M bits → 平台在 3000 条附近
          所以数据量扫 256 ~ 16384，而不是论文的 6.5 万起

用法：
    python step2_sweep.py            # 全量 7 个数据量
    python step2_sweep.py --quick    # 只跑一个最小配置，验证脚本

输出：
    out/sweep.csv    每个数据量的结果
    out/sweep.png    数据量 vs 总记忆量 曲线
"""
import os
import csv
import sys
import time
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from memlib import TinyGPT, make_random_data, code_length_bits, info_per_seq

# ---------------- 论文一致的参数 ----------------
VOCAB = 2048
SEQ = 64
STEPS = 20000
BATCH = 256
LR = 1e-3
SEED = 0

# ---------------- 等比缩小后的模型 ----------------
LAYER = 2
DIM = 128
HEAD = 4

# ---------------- 数据量网格 ----------------
SWEEP = [1024, 2048, 4096, 8192]
QUICK = [512]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def train_one(n_seq, steps, device, log=True):
    """在 n_seq 条随机数据上训练一个模型，返回（记忆量, 编码长度, 参数量, 耗时）。"""
    torch.manual_seed(SEED)

    data = make_random_data(n_seq, SEQ, VOCAB, seed=1000 + n_seq)
    model = TinyGPT(VOCAB, SEQ, LAYER, DIM, HEAD).to(device)
    n_params = model.n_params()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # 固定的验证集，用来判断是否训练到饱和
    probe = make_random_data(512, SEQ, VOCAB, seed=7).to(device)

    data_dev = data.to(device)
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, n_seq, (min(BATCH, n_seq),), device=device)
        x = data_dev[idx]
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                               x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % 200 == 0:
            hist.append(float(loss.detach()))

    cost = time.time() - t0

    per = code_length_bits(model, data_dev)
    total_bits = n_seq * (info_per_seq(SEQ, VOCAB) - per)
    probe_bits = code_length_bits(model, probe)

    if log:
        print('  N=%-6d 参数=%-8s 训练集编码=%7.2f bits  总记忆=%10.0f bits  '
              '(%.1f bit/参数)  验证集=%7.2f  耗时 %.1f 分钟'
              % (n_seq, format(n_params, ','), per, total_bits,
                 total_bits / n_params, probe_bits, cost / 60))

    return dict(n_seq=n_seq, n_params=n_params, code_len=per,
                total_bits=total_bits, bits_per_param=total_bits / n_params,
                probe_bits=probe_bits, minutes=cost / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='快速验证：小数据量 + 少步数')
    ap.add_argument('--steps', type=int, default=STEPS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    grid = QUICK if args.quick else SWEEP
    steps = 300 if args.quick else args.steps

    print('=' * 78)
    print('第 2 步：随机数据上的模型容量')
    print('=' * 78)
    print('设备 %s | V=%d S=%d | %d 层 d=%d | 每条数据 %.0f bits | 每个模型 %d 步'
          % (device, VOCAB, SEQ, LAYER, DIM, info_per_seq(SEQ, VOCAB), steps))
    print('数据量网格：%s\n' % grid)

    rows = []
    for n in grid:
        rows.append(train_one(n, steps, device))

    csv_path = os.path.join(OUT_DIR, 'sweep.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\n已保存 %s' % csv_path)

    if len(rows) < 2:
        print('只有 1 个点，跳过画图。')
        return

    fig, ax = plt.subplots(figsize=(8, 5.2))
    xs = [r['n_seq'] for r in rows]
    ys = [r['total_bits'] / 1e6 for r in rows]
    ax.plot(xs, ys, 'o-', color='#5B8FA8', linewidth=2.2, markersize=7,
            markerfacecolor='white', markeredgewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('训练数据量（条）', fontsize=12)
    ax.set_ylabel('总记忆量（百万 bits）', fontsize=12)
    ax.set_title('数据量越大，记忆量是否饱和？', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, 'sweep.png')
    fig.savefig(png, dpi=150, facecolor='white')
    print('已保存 %s' % png)


if __name__ == '__main__':
    main()
