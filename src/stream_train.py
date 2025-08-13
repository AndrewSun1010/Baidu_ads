# train.py
import os
import time
from datetime import date
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# ✨ 使用你上面实现的新数据管道
from stream_datasets import TrainStream, CollateWithStore
from evaluate import evaluate
from model import SASRec, CustomContrastiveLoss

import random, numpy as np
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# ----------------------------
# 配置
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

class Args():
    def __init__(self):
        self.dataset_dir   = os.path.join(parent_dir, "data")
        # 下面三个不再直接用到（保留以兼容你已有的目录结构）
        self.unitid_file   = 'ad_data'
        self.train_file    = "train.txt"
        self.test_file     = "test.txt"

        self.batch_size    = 32
        self.lr            = 1e-4
        self.maxlen        = 200
        self.hidden_units  = 1024
        self.emb_dim       = 1024
        self.num_blocks    = 2
        self.num_epochs    = 1
        self.num_heads     = 1
        self.dropout_rate  = 0.2
        self.device        = "gpu"
        self.inference_only = False
        self.state_dict_path = None  # e.g. "2025_03_27/xxx.pth"

        # 日志打印频率：每多少 step 打印一次（可按需修改）
        self.print_freq    = 100

args = Args()

# 保存超参（可选）
os.makedirs(os.path.join(args.dataset_dir), exist_ok=True)
with open(os.path.join(args.dataset_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

# ----------------------------
# 设备
# ----------------------------
device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
print("Using device:", device)
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)

# ----------------------------
# 模型
# ----------------------------
model = SASRec(args).to(device)
model.train()

# Xavier 初始化（仅对权重矩阵）
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            try:
                torch.nn.init.xavier_normal_(param)
            except Exception as e:
                print(f"{name} xavier 初始化失败: {e}")

# 可选：加载断点
if args.state_dict_path:
    try:
        state = torch.load(args.state_dict_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded state dict from {args.state_dict_path}")
    except Exception as e:
        print(f"failed loading state_dicts, pls check file path: {e}")

# ----------------------------
# 数据（流式 + memmap）
# ----------------------------
print("构建数据管道...")
dataset = TrainStream(
    dataset_dir=args.dataset_dir,
    pattern="train-*.jsonl.gz",
    maxlen=args.maxlen
)
collate = CollateWithStore(dataset_dir=args.dataset_dir)

# 先保守配置，确认稳定再把 workers/pin_memory 拉高
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    collate_fn=collate,
    num_workers=1,
    prefetch_factor=1,
    pin_memory=False,
    persistent_workers=False
)
print("数据管道就绪。")

# ----------------------------
# 损失、优化器、学习率
# ----------------------------
criterion = CustomContrastiveLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1, verbose=True)

# ----------------------------
# 训练
# ----------------------------
print("开始训练")
global_step = 0
# train_torch.py 训练循环里修改
eval_every = 1000             # 每 N step 触发一次验证
quick_eval_batches = 10       # 中途验证只跑前50个batch（加速）；想全量就设 None
best_metrics = {"recall@10": 0.0, "ndcg@10": 0.0}

for epoch in range(1, args.num_epochs + 1):
    if args.inference_only:
        break

    model.train()
    epoch_t0 = time.time()

    # 移动平均，降低打印噪声
    running_loss = 0.0
    ema_loss = None
    momentum = 0.98

    pbar = tqdm(dataloader, desc=f"epoch {epoch}")
    for batch in pbar:
        padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids = batch

        # 移动到设备
        padded_embeddings = padded_embeddings.to(device, non_blocking=True)
        padded_pos_emb    = padded_pos_emb.to(device, non_blocking=True)
        padded_neg_emb    = padded_neg_emb.to(device, non_blocking=True)
        pad_mask          = pad_mask.to(device, non_blocking=True)
        ad_ids            = ad_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 前向 & 反向
        logits = model(padded_embeddings, pad_mask, padded_pos_emb, padded_neg_emb)
        loss = criterion(logits, padded_pos_emb, pad_mask, ad_ids)

        loss.backward()
        optimizer.step()

        # 统计
        global_step += 1
        l = loss.item()
        running_loss += l
        ema_loss = l if ema_loss is None else (momentum * ema_loss + (1 - momentum) * l)

        # 仅每 print_freq 步刷新一次文本日志，减少 stdout 压力
        if global_step % args.print_freq == 0:
            avg = running_loss / args.print_freq
            tqdm.write(f"[epoch {epoch} step {global_step}] loss={avg:.5f} (ema={ema_loss:.5f}) lr={scheduler.get_last_lr()[0]:.6f}")
            running_loss = 0.0

        # 进度条上显示平滑 loss
        pbar.set_postfix(loss=(ema_loss if ema_loss is not None else l), lr=scheduler.get_last_lr()[0])

        # —— 中途评估（快速）——
        if (global_step % eval_every == 0):
            quick_metrics = evaluate(model, args, device, max_batches=quick_eval_batches)
            print(f"\n[QuickEval @ step {global_step}] " +
                  ", ".join([f"{k}={v:.4f}" for k, v in quick_metrics.items()]))
        model.train()  # 恢复训练模式
    # 学习率衰减
    scheduler.step()
    print(f"Epoch {epoch} done in {time.time()-epoch_t0:.1f}s, current lr: {scheduler.get_last_lr()[0]:.6f}")

    # 保存
    today = date.today()
    day = today.strftime("%Y_%m_%d")
    folder = os.path.join(parent_dir,'output',day)
    os.makedirs(folder, exist_ok=True)

    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    torch.save(model.state_dict(), os.path.join(folder, fname))

print("Done")
