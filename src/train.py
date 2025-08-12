# train_torch.py
import os
import time
from datetime import date
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import *          # 需为 PyTorch 版本/兼容
from model import *          # 需包含 torch.nn.Module 的 SASRec 与 CustomContrastiveLoss
import random, numpy as np
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# ----------------------------
# 配置
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
class Args():
    def __init__(self):
        self.dataset_dir = os.path.join(parent_dir, "data") # root directory containing the datasets
        self.unitid_file = 'ad_data'
        self.train_file = "train.txt"
        self.test_file = "test.txt"
        self.test_gt_file = "test_gt.txt"
        self.batch_size = 32
        self.lr = 1e-4
        self.maxlen = 200
        self.hidden_units = 1024
        self.emb_dim = 1024
        self.num_blocks = 2
        self.num_epochs = 1
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.device = "gpu"        # 与原保持一致；下面会映射到 torch 设备
        self.inference_only = False
        self.state_dict_path = None  # e.g. "2025_03_27/xxx.pth"

args = Args()

# 保存超参
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
# Xavier 初始化（对可训练权重做合理判断）
with torch.no_grad():
    for name, param in model.named_parameters():
        if param.dim() >= 2:   # 只对权重矩阵做 xavier，避免偏置报错
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
# 数据
# ----------------------------
print("开始加载训练数据...")
dataset = TrainDataset(args)  # 确保 __getitem__/collate_fn 返回 torch.Tensor
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    collate_fn=dataset.collate_fn,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
print("数据加载完成")

# ----------------------------
# 损失、优化器、学习率
# ----------------------------
criterion = CustomContrastiveLoss().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.98)
)

# 与 Paddle 的 ExponentialDecay(gamma=0.96) 对齐
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1, verbose=True)

# ----------------------------
# 训练
# ----------------------------
best_val_ndcg, best_val_hr = 0.0, 0.0
best_test_ndcg, best_test_hr = 0.0, 0.0

print("开始训练")
global_step = 0

for epoch in range(1, args.num_epochs + 1):
    if args.inference_only:
        break

    model.train()
    epoch_t0 = time.time()

    pbar = tqdm(dataloader)
    for batch in pbar:
        padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids = batch

        # 移动到设备
        padded_embeddings = padded_embeddings.to(device, non_blocking=True)
        padded_pos_emb     = padded_pos_emb.to(device, non_blocking=True)
        padded_neg_emb     = padded_neg_emb.to(device, non_blocking=True)
        pad_mask           = pad_mask.to(device, non_blocking=True)
        ad_ids             = ad_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 直接 FP32 前向 & 反向
        logits = model(padded_embeddings, pad_mask, padded_pos_emb, padded_neg_emb)
        loss = criterion(logits, padded_pos_emb, pad_mask, ad_ids)

        loss.backward()
        optimizer.step()

        global_step += 1
        pbar.set_description(f"epoch {epoch} | loss {loss.item():.5f} | lr {scheduler.get_last_lr()[0]:.6f}")

    # 学习率衰减
    scheduler.step()
    print(f"Epoch {epoch} done in {time.time()-epoch_t0:.1f}s, current lr: {scheduler.get_last_lr()[0]:.6f}")

    # 保存
    today = date.today()
    day = today.strftime("%Y_%m_%d")
    folder = os.path.join(parent_dir, day)
    os.makedirs(folder, exist_ok=True)

    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    torch.save(model.state_dict(), os.path.join(folder, fname))

print("Done")

