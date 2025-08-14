# evaluate.py
import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from build_vector_store import load_index, l2_normalize, search_topk
import faiss  # 可选：如果你需要设置 nprobe 等参数

# —— 索引缓存，避免每次评估都重新读盘 ——
_INDEX_CACHE = {"path": None, "index": None}

def _get_faiss_index(index_path: str, device: int = 0, nprobe: Optional[int] = None):
    global _INDEX_CACHE
    if _INDEX_CACHE["index"] is None or _INDEX_CACHE["path"] != index_path:
        _INDEX_CACHE["index"] = load_index(index_path, device=device)
        _INDEX_CACHE["path"] = index_path
        # 若是 IVF/IVF-PQ，可以设置 nprobe 提升召回；Flat 索引不生效也没关系
        if nprobe is not None:
            try:
                faiss.GpuParameterSpace().set_index_parameter(_INDEX_CACHE["index"], "nprobe", int(nprobe))
            except Exception:
                pass
    return _INDEX_CACHE["index"]

@torch.no_grad()
def evaluate(model,
             args,
             device: torch.device,
             max_batches: Optional[int] = None,
             sample_users: int = 100,
             faiss_index_path: Optional[str] = "./faiss_idx/ads_flat_ip.faiss",
             nprobe: Optional[int] = None):
    """
    从测试集抽样 sample_users 条序列：
      1) 用模型得到每条序列“最后一步”的表示向量（[B, D]）
      2) L2 归一化后在 FAISS 上检索 Top-K（K=10）
      3) 与每条序列的“最后一个真实广告ID（映射后）”对比，计算 HitRate@10 / NDCG@10

    要求：
    - 测试集返回的 ad_ids（或等价字段）与建库索引使用**相同的映射（1开始）**
    - Collate 需能提供：padded_embeddings [B,S,D]、pad_mask [B,S]、ad_ids [B,S]（或能得到最后一个真实id）
    """
    assert faiss_index_path is not None and len(faiss_index_path) > 0, "faiss_index_path 不能为空"

    # 1) 构造测试集 DataLoader（与你的 stream_datasets 对齐）
    from stream_datasets import CollateWithStore
    from stream_datasets import TestStream
    eval_ds = TestStream(dataset_dir=args.dataset_dir, pattern="test-*.jsonl.gz", maxlen=args.maxlen)


    collate = CollateWithStore(dataset_dir='/root/autodl-tmp/code_pt/data')  # 若无 mode 参数，去掉它
    eval_loader = DataLoader(
        eval_ds,
        batch_size=min(args.batch_size, 64),
        collate_fn=collate,
        num_workers=1,
        pin_memory=False,
        shuffle=False
    )

    # 2) 载入（或复用缓存的）FAISS 索引到 GPU
    device_index = 0 if torch.cuda.is_available() and device.type == "cuda" else None
    index = _get_faiss_index(faiss_index_path, device=device_index, nprobe=nprobe)

    # 3) 遍历样本直到攒够 sample_users
    K = 10
    hits, ndcgs = [], []
    processed = 0
    batches_done = 0

    for batch in eval_loader:
        # 兼容你的 collate 返回（训练返回的是：padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids）
        if len(batch) == 5:
            padded_embeddings, _pos, _neg, pad_mask, ad_ids = batch
        else:
            # 若 eval collate 简化为三元组（padded_embeddings, pad_mask, ad_ids）
            padded_embeddings, pad_mask, ad_ids = batch

        padded_embeddings = padded_embeddings.to(device, non_blocking=True)  # [B,S,D]
        pad_mask          = pad_mask.to(device, non_blocking=True)          # [B,S]
        ad_ids            = ad_ids.to(device, non_blocking=True)            # [B,S]（映射后的ID，1开始）

        # 3.1 取“最后一步”的模型表示：logits = model.log2feats(...); 取最后有效位置
        #     如果你的 SASRec 有 predict 返回最后一步，可直接用；这里使用 log2feats 最稳妥
        log_feats = model(padded_embeddings, pad_mask)            # [B,S,D]
        B = log_feats.size(0)
        # 选出每条序列的最后一步隐藏向量
        q = log_feats[:, -1, :]              # [B,D]
        # ground-truth 最后一个 ID（和索引同一映射口径）
        gt = ad_ids[:, -1]                # [B]

        # 3.2 FAISS 检索
        q_np = q.detach().float().cpu().numpy()                              # [B,D]
        # 归一化，确保用余弦（建库时也应做了 normalize）
        q_np = l2_normalize(q_np)
        _scores, topk_ids = search_topk(index, q_np, topk=K, normalize=False)  # 已手动归一化

        # 3.3 计算指标（逐条）
        topk_ids = torch.as_tensor(topk_ids, device=device)                  # [B,K]
        for i in range(B):
            if processed >= sample_users:
                break
            gi = int(gt[i].item())
            topk_i = topk_ids[i].tolist()
            # Hit@10
            hit = 1.0 if gi in topk_i else 0.0
            hits.append(hit)
            # NDCG@10
            if hit > 0:
                rank = topk_i.index(gi) + 1  # 1-based
                ndcg = 1.0 / math.log2(rank + 1)
            else:
                ndcg = 0.0
            ndcgs.append(ndcg)
            processed += 1

        batches_done += 1
        if max_batches is not None and batches_done >= max_batches:
            break
        if processed >= sample_users:
            break

    # 4) 汇总
    if processed == 0:
        return {"hitrate@10": 0.0, "ndcg@10": 0.0}

    metrics = {
        "hitrate@10": float(np.mean(hits)),
        "ndcg@10": float(np.mean(ndcgs)),
        "num_eval": processed
    }
    return metrics
