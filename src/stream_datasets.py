# datasets.py  —— 训练时使用
import os, json, glob, gzip, orjson, numpy as np, torch
from torch.utils.data import IterableDataset

# -----------------------------
# 只读向量库（memmap）
# -----------------------------
class AdVectorStore:
    def __init__(self, dataset_dir: str):
        meta = json.load(open(os.path.join(dataset_dir, "ad_meta.json")))
        self.N, self.D = meta["N"], meta["D"]
        self.mmap = np.memmap(os.path.join(dataset_dir, "ad_vectors.fp16.memmap"),
                              dtype=np.float16, mode="r", shape=(self.N, self.D))

    def get(self, idxs: np.ndarray) -> np.ndarray:
        # [L] or [K] -> [L,D] float32（懒转换，减少内存峰值）
        return self.mmap[idxs].astype(np.float32, copy=False)

# -----------------------------
# 流式读取 idx 分片
# -----------------------------
class TrainStream(IterableDataset):
    """
    读取 preprocess 后的 train-*.jsonl.gz
    每次 yield: np.ndarray[int64]（长度 Li）
    """
    def __init__(self, dataset_dir: str, pattern="train-*.jsonl.gz", maxlen=200):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
        self.maxlen = maxlen

    def __iter__(self):
        for fp in self.files:
            with gzip.open(fp, "rb") as f:
                for line in f:
                    obj = orjson.loads(line)
                    idx = np.array(obj["idx"], dtype=np.int64)
                    if idx.size <= 1: 
                        continue
                    yield idx[-self.maxlen:]

class TestStream(IterableDataset):
    """
    读取 preprocess 后的 test-*.jsonl.gz
    每次 yield: np.ndarray[int64]（长度 Li）
    """
    def __init__(self, dataset_dir: str, pattern="test-*.jsonl.gz", maxlen=200):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.files = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
        self.maxlen = maxlen

    def __iter__(self):
        for fp in self.files:
            with gzip.open(fp, "rb") as f:
                for line in f:
                    obj = orjson.loads(line)
                    idx = np.array(obj["idx"], dtype=np.int64)
                    if idx.size <= 1: 
                        continue
                    yield idx[-self.maxlen:]
# -----------------------------
# collate：一次性取向量 + 负采样 + padding(0)
# -----------------------------
class CollateWithStore:
    def __init__(self, dataset_dir: str):
        self.store = AdVectorStore(dataset_dir)

    def __call__(self, batch_ids):
        """
        batch_ids: List[np.ndarray]，每个长度 Li（>=2）
        返回：
          qk, pos, neg, mask, ad_ids  （与原接口一致）
        """
        B = len(batch_ids)
        D = self.store.D

        # 因为对于最长的序列[:-1]为训练样本，[1:]为正样本
        max_len = max(len(x) for x in batch_ids) - 1
        qk  = torch.zeros(B, max_len, D, dtype=torch.float32)
        pos = torch.zeros(B, max_len, D, dtype=torch.float32)
        neg = torch.zeros(B, max_len, D, dtype=torch.float32)
        mask= torch.zeros(B, max_len,    dtype=torch.bool)
        ad_ids = torch.zeros(B, max_len, dtype=torch.long)

        N = self.store.N

        for i, ids in enumerate(batch_ids):
            L = len(ids) - 1
            if L <= 0: continue
            mask[i, -L:] = True
            ad_ids[i, -L:] = torch.from_numpy(ids[1:])

            # 正样本向量（一次性取）
            qk_np  = self.store.get(ids[:-1])  # [L,D]
            pos_np = self.store.get(ids[ 1:])  # [L,D]

            # 负采样（简单、允许重复；如需排除可向量化优化）
            neg_ids = np.random.randint(0, N, size=L, dtype=np.int64)
            neg_np  = self.store.get(neg_ids)  # [L,D]

            qk[i,  -L:] = torch.from_numpy(qk_np)
            pos[i, -L:] = torch.from_numpy(pos_np)
            neg[i, -L:] = torch.from_numpy(neg_np)

        return qk, pos, neg, mask, ad_ids

if __name__ == "__main__":
    # 简单测试
    dataset = TrainStream(dataset_dir="/root/autodl-tmp/code_pt/data", pattern="train-*.jsonl.gz", maxlen=200)
    collate = CollateWithStore(dataset_dir="/root/autodl-tmp/code_pt/data")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)
    for qk, pos, neg, mask, ad_ids in loader:
        print(qk.shape, pos.shape, neg.shape, mask.shape, ad_ids.shape)
        break
    # 输出
    # torch.Size([4, 23, 1024]) torch.Size([4, 23, 1024]) torch.Size([4, 23, 1024]) torch.Size([4, 23]) torch.Size([4, 23])