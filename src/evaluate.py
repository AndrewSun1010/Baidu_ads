# train_torch.py 顶部或末尾添加
import torch
import os
from torch.utils.data import DataLoader
from collections import defaultdict
from stream_datasets import TestStream, CollateWithStore

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dataset_dir = os.path.join(parent_dir, "data")


collate = CollateWithStore(dataset_dir=os.path.join(parent_dir, "data"))


@torch.no_grad()
def compute_recall_ndcg(scores, gt_index, ks=[10]):
    """
    scores:   [B, K]
    gt_index: [B]  取值范围[0, K-1]
    return:   {'recall@5':..., 'ndcg@10':...}
    """
    B, K = scores.shape
    # 排名（按分数降序的候选索引）
    top_order = torch.argsort(scores, dim=1, descending=True)  # [B, K]

    # 每个样本的命中rank（0-based）
    arange_k = torch.arange(K, device=scores.device).unsqueeze(0).expand(B, K)  # [B,K]
    # 找到 gt_index 在排序中的位置
    is_gt = (top_order == gt_index.unsqueeze(1))                                 # [B,K]
    gt_rank = torch.where(is_gt, arange_k, torch.full_like(arange_k, K)).min(dim=1).values  # [B]

    out = {}
    for k in ks:
        hit = (gt_rank < k).float()                         # [B]
        recall_k = hit.mean().item()

        # DCG：只有一个正样本，命中时 dcg = 1/log2(rank+2)
        dcg = torch.where(hit.bool(), 1.0 / torch.log2(gt_rank.float() + 2.0), torch.zeros_like(hit))
        idcg = torch.ones_like(dcg)  # 单个正样本的 IDCG=1
        ndcg_k = (dcg / idcg).mean().item()

        out[f"recall@{k}"] = recall_k
        out[f"ndcg@{k}"] = ndcg_k
    return out


@torch.no_grad()
def evaluate(model, args, device, max_batches=None):
    """
    max_batches: 仅评估前N个batch（可选，加速中途验证）
    需要 TestDataset(args).collate_fn 返回 (padded_embeddings, pad_mask, gt_index, gt_embeddings)
    """
    model.eval()
    test_set = TestStream(
        dataset_dir=args.dataset_dir,
        pattern="test-*.jsonl.gz",
        maxlen=args.maxlen
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=1,
        pin_memory=False,
        persistent_workers=False
    )

    meter = defaultdict(float)
    n = 0

    for b_idx, batch in enumerate(test_loader):
        padded_embeddings, padded_pos_emb, padded_neg_emb, pad_mask, ad_ids = batch
        padded_embeddings = padded_embeddings.to(device, non_blocking=True)
        pad_mask          = pad_mask.to(device, non_blocking=True)
        item_embs         =  padded_pos_emb.to(device, non_blocking=True)  # 与历史序列比较
        B = item_embs.size(0)
        K = item_embs.size(1)
        gt_index = torch.full(
            (B,), fill_value=K - 1, dtype=torch.long, device=device
        )

        # 得分：B x K
        # 若你已有 model.predict(...)，用它即可；否则用上面的 score_candidates
        scores = model.predict(padded_embeddings, pad_mask, item_embs)

        res = compute_recall_ndcg(scores, gt_index, ks=[10])
        for k, v in res.items():
            meter[k] += v * padded_embeddings.size(0)
        n += padded_embeddings.size(0)

        if (max_batches is not None) and (b_idx + 1 >= max_batches):
            break

    # 汇总
    for k in list(meter.keys()):
        meter[k] /= max(n, 1)

    return dict(meter)
