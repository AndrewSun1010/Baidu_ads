import numpy as np
import faiss
from build_vector_store import load_index, search_topk, l2_normalize

# 读索引并搬GPU
index = load_index("./faiss_idx/ads_flat_ip.faiss", device=0)

# sasrec_last is [B, D] (未归一化)，若建库时用了 --normalize，这里也需要归一化
sasrec_last = np.random.randn(4, index.d).astype(np.float32)
sasrec_last = l2_normalize(sasrec_last)
scores, topk_ids = search_topk(index, sasrec_last, topk=10, normalize=True)

# topk_ids 中就是“映射后的连续ID(从1开始)”，可直接与 TestDataset 的 gt_id 对齐评估
print(topk_ids)
