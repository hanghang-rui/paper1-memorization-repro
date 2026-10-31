# -*- coding: utf-8 -*-
r"""
step1_sanity.py —— 第一步：验证"记忆量"的度量函数写对了没有

对应论文 2.3 节：用 −log2 p(x) 当作"模型编码这条数据所需的比特数"。

判据只有两条：
  1) 没训练过的随机模型：每条样本的编码长度 ≈ (S-1) * log2(V)
  2) 在某几条样本上过拟合训练过的模型：编码长度应明显低于上面那个数

两个数都对上，说明你的度量是对的，可以进 3.2 节的容量实验。

运行：  D:\python\.venv-mem\Scripts\python.exe step1_sanity.py
"""
import math  #后面算log2.ln2转换要用
import torch  #深度学习框架
from memlib import TinyGPT, make_random_data, code_length_bits, info_per_seq, train  #分别为：最小GPT模型类、生成均匀随机token数据、算‘每条序列的编码长度（长度）’、算理论值、编著GPT训练循环

#超参数定义
# ---------------- 小配置，几秒钟就能跑完 ----------------
VOCAB = 64  #词表大小：64种token  每个token都是64选1---猜中它需要log2(64)=6 bits 信息
SEQ = 32   #序列长度 表示每条样本是一串32个整数的token,模型一次就看这么长的窗口
LAYER = 2
DIM = 64
HEAD = 4
SEED = 0

torch.manual_seed(SEED)  #设置随机数
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('=' * 62)
print('第一步：度量函数的 sanity check')

print('=' * 62)
print(f'设备：{device}    词表 V={VOCAB}    序列长 S={SEQ}')
print(f'每条样本的原始信息量 = (S-1) * log2(V) = {info_per_seq(SEQ, VOCAB):.1f} bits\n')  #VO调用了memlib.py里面的函数info_per_seq

# ---------------- 情况 1：没训练过的模型 ----------------
model = TinyGPT(VOCAB, SEQ, LAYER, DIM, HEAD).to(device)   #创建模型
print(f'模型参数量：{model.n_params():,}')  #打印参数量，定义在memlib L53-54

probe = make_random_data(200, SEQ, VOCAB, seed=1).to(device)  #造’探针‘数据
bits_untrained = code_length_bits(model, probe)  #本质就是在算bit数，本实验就是在用bit数来判断模型对数据的熟练程度：bit数越小表示越熟悉（背过），反之不熟悉（若完全没学过，则bits=186(理论值））
print(f'\n[情况1] 未训练的模型')
print(f'  每条样本编码长度 = {bits_untrained:8.2f} bits')
print(f'  理论值（瞎猜）   = {info_per_seq(SEQ, VOCAB):8.2f} bits')
print(f'  偏差             = {bits_untrained - info_per_seq(SEQ, VOCAB):+8.2f} bits')

# ---------------- 情况 2：在少量样本上过拟合 ---------------- #过拟合就是’死记硬背‘  这里一定会过拟合发原因：1。数据是随机的 2.只有32条样本（模型11万条参数，背32条绰绰有余）
print(f'\n[情况2] 在 32 条样本上过拟合训练')
mem_set = make_random_data(32, SEQ, VOCAB, seed=2)  #创建随机数据
final_loss = train(model, mem_set, steps=1500, batch=32, lr=3e-3,
                   log_every=500, device=device)

bits_trained = code_length_bits(model, mem_set.to(device))
print(f'\n  记住的这 32 条样本，编码长度 = {bits_trained:8.2f} bits')

probe2 = make_random_data(200, SEQ, VOCAB, seed=3).to(device)
bits_heldout = code_length_bits(model, probe2)
print(f'  没见过的样本，编码长度       = {bits_heldout:8.2f} bits')

# ---------------- 判定 ----------------
print('\n' + '=' * 62)
print('判定')
print('=' * 62)
theory = info_per_seq(SEQ, VOCAB)  #确定基准线（理论值=186bits）
ok1 = abs(bits_untrained - theory) < 0.05 * theory
ok2 = bits_trained < 0.5 * theory

print(f'  未训练模型 ≈ 理论值     ：{bits_untrained:.1f} vs {theory:.1f}   -> {"通过" if ok1 else "不通过"}')
print(f'  过拟合后明显低于理论值   ：{bits_trained:.1f} < {theory * 0.5:.1f}   -> {"通过" if ok2 else "不通过"}')
print(f'  训练集与未见样本的差距   ：{bits_heldout - bits_trained:+.1f} bits（差值越大说明越像"背下来"）')

if ok1 and ok2:
    print('\n结论：度量函数正确，可以进入 3.2 节的容量实验（step2_sweep.py）。')
else:
    print('\n结论：有问题。最常见原因是把 log 底数写成了 e（应当用 log2），')
    print('      或者模型没有真正过拟合（加训练步数再试）。')
