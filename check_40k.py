# -*- coding: utf-8 -*-
"""
check_40k.py —— 单点验证：N=8192、40000 步

目的：看「训练步数加倍后，容量是否继续上升」。
  · 若 bit/参数 从 2.3 升到 2.8–3.0 → 说明平台受训练量限制，加步数就能逼近论文的 3.5
  · 若仍在 2.3 附近       → 说明小模型本身的容量就低于论文值，也是有价值的发现

不影响 step2_sweep.py 和已有的 out/sweep.csv
"""
import os
import time
import math

import torch
import torch.nn.functional as F

from memlib import TinyGPT, make_random_data, code_length_bits, info_per_seq

VOCAB, SEQ = 2048, 64
LAYER, DIM, HEAD = 2, 128, 4
N_SEQ = 8192
STEPS = 40000
BATCH = 256
LR = 1e-3
SEED = 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')


def main():
    torch.manual_seed(SEED)
    data = make_random_data(N_SEQ, SEQ, VOCAB, seed=1000 + N_SEQ).to(DEVICE)
    model = TinyGPT(VOCAB, SEQ, LAYER, DIM, HEAD).to(DEVICE)
    n_params = model.n_params()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    print('=' * 70)
    print('单点验证：N=%d，%d 步（正式扫描用的是 20000 步）' % (N_SEQ, STEPS))
    print('设备 %s | 参数 %s | 每条数据 %.0f bits'
          % (DEVICE, format(n_params, ','), info_per_seq(SEQ, VOCAB)))
    print('=' * 70)

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
    print('  bit/参数 = %.2f   （20000 步时是 2.35，论文是 3.5–3.6）'
          % (total / n_params))
    print('  耗时 %.1f 分钟' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
