# -*- coding: utf-8 -*-
"""
memlib.py —— 复现《How much can language models memorize?》用到的公共代码。

只实现论文第 2.3 节 + 第 3.2 节需要的东西：
  1. 造随机 token 数据（均匀采样，没有可学的规律）
  2. 一个最小的 GPT 式 Transformer
  3. 用 −log2 p 估计"模型记住多少比特"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 模型
class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, seq_len, n_layer=2, d=64, n_head=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.seq_len = seq_len
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, idx):
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        mask = self.mask[:S, :S]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())  #算模型全部参数（权重+偏置）不包含缓冲区


# ---------------------------------------------------------------- 数据
def make_random_data(n_seq, seq_len, vocab, seed=0):
    """均匀随机 token：每条序列的信息量恰好是 seq_len * log2(vocab)。"""
    g = torch.Generator().manual_seed(seed)  #torch,Generator()表示创建一个独立的随机数生成器对象（不和全局那个共用状态）manuai_seed(seed)表示给它设种子，只要seed相同，生成的数据就完全相同--这是数据可复现的保证
    return torch.randint(0, vocab, (n_seq, seq_len), generator=g) #torch.randint(0,vocab,形状)：从[0,vocab)里面军需抽取整数， (n_seq, seq_len)表示输出形状B*S,generator=g表示制订用刚才那个独立说生成器抽数，而不是全局随机数


# ---------------------------------------------------------------- 度量
@torch.no_grad()
def code_length_bits(model, data, batch=512):
    """返回每条序列被模型编码所需的比特数（−log2 p(x) 的样本均值）。

    模型完全没学到 → 每条 ≈ (S-1) * log2(V)
    模型完全背下来 → 接近 0
    """
    model.eval()
    total = 0.0
    n = 0
    for i in range(0, len(data), batch):  #按照512条切块
        x = data[i:i + batch]  #取一个批次
        logits = model(x)  #每个位置对64种token各给一个分数
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        tok = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        total += float((-tok.sum(dim=1) / math.log(2)).sum())  #log_softmax给的是自然对数，除以ln2就换算成比特
        n += x.shape[0]
    return total / n


def info_per_seq(seq_len, vocab):
    """一条序列的原始信息量（比特）；预测 S-1 个 token。"""
    return (seq_len - 1) * math.log2(vocab)   #log2=‘二分多少次能确定答案’  一次是/否=1bit


def memorization_bits(model, data, seq_len, vocab):
    """总记忆量 = 数据总信息量 − 模型编码总长度（比特）。"""
    per = code_length_bits(model, data)
    return len(data) * (info_per_seq(seq_len, vocab) - per), per


# ---------------------------------------------------------------- 训练
def train(model, data, steps=2000, batch=128, lr=3e-3, log_every=500, device='cuda'):
    model.to(device).train()  #把模型搬到GPU/CPU，表示开始工作了
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)   #’请一个教练，根据错误分去拧模型的参数’，1r=3e-3是学习率
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  #’给教练定节奏：先大步快调，后期小步精调‘
    data = data.to(device)
    n = len(data)  #数一下一共几道题
    last = None
    for step in range(1, steps + 1):  #循环step遍
        idx = torch.randint(0, n, (min(batch, n),), device=device)  #随机抽题号
        x = data[idx]  #把抽到的题拿出来
        logits = model(x)   #让模型做一遍（猜每个位置的下一个）
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               x[:, 1:].reshape(-1))  #‘对答案：算出总共错多少（错误分）’
        opt.zero_grad(set_to_none=True)  #擦干净上次记的错账
        loss.backward()  #让错误顺着模型‘倒流’，算出每个参数该往哪调
        opt.step()  #‘教练动手：真拧一遍所有参数
        sched.step()  #把下一步的劲调小一点
        last = float(loss.detach())  #把这次错误分抄进计分板（只要数字，不带账本）
        if step % log_every == 0:  #每500遍汇报一次进度
            print('    step %5d  loss %.4f  (bits/token %.3f)'
                  % (step, last, last / math.log(2)))
    return last
