# -*- coding: utf-8 -*-
"""
check_tied.py —— 验证「词嵌入与输出层共享参数」能否解释 bit/参数 的差距

背景
----
原实现里，词嵌入（2048×128）和输出层（2048×128）是两张独立的表，
各自 262,144 个参数，合计 524,288，占总参数 929,280 的 56%。
但它们内容上几乎是同一张「字表」，等于存了两遍。

GPT-2 等模型的做法是把输出层直接绑到词嵌入上（weight tying），只存一份。

本脚本只改这一点，其余全部与 check_40k.py 相同，用来做对照：

    原配置（未共享）：N=8192，40000 步 → 总记忆 2,290,173 bits，929,280 参数 → 2.46 bit/参数
    本  脚本（已共享）：N=8192，40000 步 → 参数量应降到 667,136，若总记忆不变，比例升到约 3.4

不修改 memlib.py，不覆盖任何已有结果。

运行：
    D:\\python\\.venv-mem\\Scripts\\python.exe check_tied.py
    约 13 分钟（40000 步）
"""
import os
import time

import torch
import torch.nn.functional as F

from memlib import TinyGPT, make_random_data, code_length_bits, info_per_seq

VOCAB, SEQ = 2048, 64
LAYER, DIM, HEAD = 2, 128, 4
N_SEQ = 8192
STEPS = 40000          # 与 check_40k.py 保持一致，唯一变量就是「是否共享」
BATCH = 256
LR = 1e-3
SEED = 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    torch.manual_seed(SEED)

    data = make_random_data(N_SEQ, SEQ, VOCAB, seed=1000 + N_SEQ).to(DEVICE)
    model = TinyGPT(VOCAB, SEQ, LAYER, DIM, HEAD).to(DEVICE)

    n_before = model.n_params()
    # ---- 唯一改动：输出层直接复用词嵌入的那张表 ----
    model.head.weight = model.tok.weight
    n_after = model.n_params()

    print('=' * 72)
    print('对照实验：权重共享（weight tying）')
    print('=' * 72)
    print('设备 %s | N=%d | %d 步 | 每条数据 %.0f bits'
          % (DEVICE, N_SEQ, STEPS, info_per_seq(SEQ, VOCAB)))
    print('参数：共享前 %s → 共享后 %s（减少 %s）'
          % (format(n_before, ','), format(n_after, ','),
             format(n_before - n_after, ',')))
    print('-' * 72)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, N_SEQ, (BATCH,), device=DEVICE)
        x = data[idx]
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 5000 == 0:
            print('  步 %6d  loss %.4f  已用 %.1f 分钟'
                  % (step, float(loss.detach()), (time.time() - t0) / 60))

    per = code_length_bits(model, data)
    total = N_SEQ * (info_per_seq(SEQ, VOCAB) - per)
    print('\n结果：')
    print('  训练集每条剩余编码 = %.2f bits（满分 %.0f）' % (per, info_per_seq(SEQ, VOCAB)))
    print('  总记忆 = %.0f bits' % total)
    print('  bit/参数（共享后参数量）= %.2f' % (total / n_after))
    print('  bit/参数（沿用旧口径）  = %.2f' % (total / n_before))
    print('  耗时 %.1f 分钟' % ((time.time() - t0) / 60))
    print('-' * 72)
    print('对照：未共享时 2.46（总记忆 2,290,173 / 929,280）；论文 3.5–3.6')
    print('若上面第一个数升到 3.2–3.5，说明差距主要来自参数统计口径，而非训练不足')


if __name__ == '__main__':
    main()
