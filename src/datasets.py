import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

# -----------------------------
# 工具函数：读 unit 向量
# -----------------------------
def read_data(f, unitid_data):
    with open(f, 'r') as fh:
        for line in tqdm(fh):
            parts = line.strip().split('\t')
            ad_id = int(parts[0])
            # embedding = list(map(np.float32, parts[2].split(',')))
            # unitid_data[ad_id] = {'embedding': embedding}
            # embedding = np.fromstring(parts[2], sep=',', dtype=np.float16)  # 存储用 fp16
            # 直接用string存储，避免转换
            embedding = parts[2]
            unitid_data[ad_id] = embedding  # 不要再包一层 dict


def safe_process_file(f, unitid_data):
    try:
        read_data(f, unitid_data)
    except Exception as e:
        print(e)

# -----------------------------
# 训练集
# -----------------------------
class TrainDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.unitid_data, self.lenth_unit_data = self.load_unit()
        self.unit_keys = np.array(list(self.unitid_data.keys()), dtype=np.int64)
        self.train_data = []

        cnt_null_unit = 0
        with open(f"{args.dataset_dir}/{args.train_file}", 'r') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                # ad_ids = list(map(int, parts[1].split()))[:-1]  # 最后一位暂作测试用，不用于训练
                ad_ids = list(map(int, parts[1].split())) # 有专门的test集
                ad_filter_ids = []
                for ad_id in ad_ids:
                    if ad_id in self.unitid_data:
                        ad_filter_ids.append(ad_id)
                    else:
                        cnt_null_unit += 1
                        # print(f"{ad_id} not in unit_map")
                self.train_data.append({'ad_ids': ad_filter_ids})
        print(f"train.txt loaded sucessfully ,{len(self.train_data)},{cnt_null_unit} units which is not in unit_map")

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        sample = self.train_data[idx]
        ad_ids = sample['ad_ids']

        ad_embeddings = []
        for ad_id in ad_ids:
            if ad_id in self.unitid_data:
                # ad_embeddings.append(self.unitid_data[ad_id]['embedding'])
                # ad_embeddings.append(self.unitid_data[ad_id])
                # 将字符串转换为向量
                ad_embeddings.append(np.fromstring(self.unitid_data[ad_id], sep=',', dtype=np.float32))
            else:
                print(f"{ad_id} not in unit_map")
        lenth = len(ad_embeddings)
        return ad_embeddings[lenth - self.args.maxlen:], ad_ids[lenth - self.args.maxlen:]

    def load_unit(self):
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        # root_dir = "data/data322235/w_data/1w_unitid_title_emb"  # 按需修改
        # file_list = [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files]
        file_list = [os.path.join(self.args.dataset_dir, self.args.unitid_file)]
        unitid_data = {}
        for f in file_list:
            safe_process_file(f, unitid_data)
        lenth_unit_data = len(unitid_data)
        if 0 not in unitid_data:
            print("check right: ad_id 0 can be pad id")
        print(f"length unitid {lenth_unit_data}")
        print(f"{time.time() - st}")
        return unitid_data, lenth_unit_data

    def collate_fn(self, batch):
        # batch: List[(ad_embeddings(list[np.ndarray]), ad_ids(list[int]))]
        ad_embeddings, ad_ids = zip(*batch)

        # 以可训练长度为准：seq_len = len(emb) - 1
        seq_lens = [max(len(emb) - 1, 0) for emb in ad_embeddings]
        # 过滤掉过短的样本（可选）
        # 如果你不想丢数据，至少要在下面对 seq_len==0 做特殊处理
        max_seq_len = max(seq_lens) if len(seq_lens) > 0 else 0
        D = self.args.emb_dim

        padded_embeddings = []
        padded_pos_embs = []
        padded_neg_embs = []
        pad_mask = []
        padded_ad_ids = []

        # 建议把 keys 缓存到 __init__：
        unit_keys = self.unit_keys  # np.ndarray of all ad_ids
        num_pool = len(unit_keys)

        for emb, ids in zip(ad_embeddings, ad_ids):
            # numpy → torch 的惯用写法（避免多次复制）
            embs_np = np.asarray(emb, dtype=np.float32)              # shape [L, D]
            ids_np  = np.asarray(ids, dtype=np.int64)                # shape [L]
            L = embs_np.shape[0]
            if L <= 1:
                # 跳过或构造全 pad（这里以“全 pad”返回为例）
                padded_embeddings.append(torch.zeros(max_seq_len, D, dtype=torch.float32))
                padded_pos_embs.append(torch.zeros(max_seq_len, D, dtype=torch.float32))
                padded_neg_embs.append(torch.zeros(max_seq_len, D, dtype=torch.float32))
                pad_mask.append(torch.zeros(max_seq_len, dtype=torch.bool))
                padded_ad_ids.append(torch.zeros(max_seq_len, dtype=torch.long))
                continue

            # query/key: emb[:-1],  pos: emb[1:], ids 对齐为 ids[1:]
            qk_np  = embs_np[:-1]          # [L-1, D]
            pos_np = embs_np[1:]           # [L-1, D]
            ids_np = ids_np[1:]            # [L-1]
            seq_len = L - 1
            pad_len = max_seq_len - seq_len

            # 负采样：从 unit_keys 里抽取 seq_len 个 id，排除 pad_id(0) 与 ids_np 中的 id
            # 先构造可选池（注意：unit_keys 是 ad_id，不是索引）

            # 构建排除集合
            exclude = set(ids_np.tolist())
            # 反复采直到集齐 seq_len（简单安全；如需更快可用向量化/重采样策略）
            neg_ids = []
            pool = unit_keys
            while len(neg_ids) < seq_len:
                cand = np.random.choice(pool, size=seq_len - len(neg_ids), replace=True)
                cand = [x for x in cand if x not in exclude]
                neg_ids.extend(cand)
            neg_ids = np.asarray(neg_ids[:seq_len], dtype=np.int64)
            # 把 neg_ids 转成向量
            neg_emb_list = [np.fromstring(self.unitid_data[int(aid)], sep=',', dtype=np.float32) for aid in neg_ids]
            neg_np = np.stack(neg_emb_list, axis=0)  # [L-1, D]

            # padding：前 pad（让尾部对齐）
            if pad_len > 0:
                pad_vec = np.random.randn(pad_len, D).astype(np.float32)
                qk_np  = np.concatenate([pad_vec, qk_np],  axis=0)
                pos_np = np.concatenate([pad_vec, pos_np], axis=0)
                neg_np = np.concatenate([pad_vec, neg_np], axis=0)
                mask_t = torch.zeros(max_seq_len, dtype=torch.bool)
                mask_t[-(seq_len):] = True
                ids_t  = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                    torch.from_numpy(ids_np)], dim=0)
            else:
                mask_t = torch.ones(max_seq_len, dtype=torch.bool)
                ids_t  = torch.from_numpy(ids_np)

            padded_embeddings.append(torch.from_numpy(qk_np))   # [B,L,D]
            padded_pos_embs.append(torch.from_numpy(pos_np))    # [B,L,D]
            padded_neg_embs.append(torch.from_numpy(neg_np))    # [B,L,D]
            pad_mask.append(mask_t)                             # [B,L]
            padded_ad_ids.append(ids_t)                         # [B,L] (long)

        padded_embeddings     = torch.stack(padded_embeddings, dim=0)
        padded_pos_embeddings = torch.stack(padded_pos_embs, dim=0)
        padded_neg_embeddings = torch.stack(padded_neg_embs, dim=0)
        pad_mask              = torch.stack(pad_mask, dim=0)
        padded_ad_ids         = torch.stack(padded_ad_ids, dim=0)

        return padded_embeddings, padded_pos_embeddings, padded_neg_embeddings, pad_mask, padded_ad_ids
        # # batch 中每个元素为 (ad_embeddings(list of vec), ad_ids(list))
        # ad_embeddings, ad_ids = zip(*batch)
        # # 找到最长序列（注意这里用 emb[:-1]）
        # # max_len = max(len(emb[:-1]) for emb in ad_embeddings)
        # max_len = max(len(emb) for emb in ad_embeddings)

        # # 建议把 keys 缓存到 __init__：
        # unit_keys = self.unit_keys  # np.ndarray of all ad_ids
        # num_pool = len(unit_keys)

        # padded_embeddings = []
        # pad_mask = []
        # padded_pos_embs = []
        # padded_neg_embs = []
        # padded_ad_ids = []

        # for idx, emb in enumerate(ad_embeddings):
        #     emb_len = len(emb)
        #     ad_ids_vector = torch.tensor(ad_ids[idx][1:], dtype=torch.long)
        #     padding_len = max_len - emb_len

        #     if padding_len:
        #         padding_vector = torch.randn(padding_len, self.args.emb_dim, dtype=torch.float32)
        #         padding_ad_vector = torch.full((padding_len,), 0, dtype=torch.float32)  # pad id 0
        #         # padded_emb = torch.cat([padding_vector, torch.tensor(emb[:-1], dtype=torch.float32)], dim=0)
        #         padded_emb = torch.cat([
        #             padding_vector,
        #             torch.tensor(np.array(emb[:-1]), dtype=torch.float32)
        #         ], dim=0)
        #         padded_ad_ids_vector = torch.cat([padding_ad_vector, ad_ids_vector], dim=0)
        #     else:
        #         # padded_emb = torch.tensor(emb[:-1], dtype=torch.float32)
        #         padded_emb = torch.tensor(np.array(emb[:-1]), dtype=torch.float32)
        #         padded_ad_ids_vector = ad_ids_vector

        #     padded_ad_ids.append(padded_ad_ids_vector)

        #     mask = torch.ones(max_len, dtype=torch.float32)
        #     if padding_len:
        #         mask[:padding_len] = 0
        #     pad_mask.append(mask)
        #     padded_embeddings.append(padded_emb)

        #     # 正例（右移一位）
        #     if padding_len:
        #         # padded_pos_emb = torch.cat([padding_vector, torch.tensor(emb[1:], dtype=torch.float32)], dim=0)
        #         padded_pos_emb = torch.cat([padding_vector, torch.tensor(np.array(emb[1:]), dtype=torch.float32)], dim=0)
        #     else:
        #         padded_pos_emb = torch.tensor(np.array(emb[1:]), dtype=torch.float32)

        #     # 随机负例
        #     unit_data_map_ids = list(self.unitid_data.keys())
        #     # 注意：原实现传了 [ad_ids[idx][1:]]（list 包 list），这里保持一致
        #     random_neg_ids = self.generate_random_numbers(0, self.lenth_unit_data - 1, [ad_ids[idx][1:]], emb_len)
        #     # random_neg_emb = torch.tensor(
        #     #     [self.unitid_data[unit_data_map_ids[i]]['embedding'] for i in random_neg_ids],
        #     #     dtype=torch.float32
        #     # )
        #     random_neg_emb = torch.tensor(
        #         [np.fromstring(self.unitid_data[unit_data_map_ids[i]], sep=',', dtype=np.float32) for i in random_neg_ids],
        #         dtype=torch.float32
        #     )
        #     if padding_len:
        #         padded_neg_emb = torch.cat([padding_vector, random_neg_emb], dim=0)
        #     else:
        #         padded_neg_emb = random_neg_emb

        #     padded_pos_embs.append(padded_pos_emb)
        #     padded_neg_embs.append(padded_neg_emb)

        # padded_embeddings = torch.stack(padded_embeddings, dim=0)      # [B, L, D]
        # pad_mask = torch.stack(pad_mask, dim=0)                        # [B, L]
        # padded_pos_embeddings = torch.stack(padded_pos_embs, dim=0)    # [B, L, D]
        # padded_neg_embeddings = torch.stack(padded_neg_embs, dim=0)    # [B, L, D]
        # padded_ad_ids = torch.stack(padded_ad_ids, dim=0)              # [B, L]

        # return padded_embeddings, padded_pos_embeddings, padded_neg_embeddings, pad_mask, padded_ad_ids

    def generate_random_numbers(self, start, end, exceptions, count):
        """
        生成 count 个在 [start, end] 范围内的随机数，但不能是 exceptions 列表中的值。
        这里保持与原 Paddle 代码一致：exceptions 可能是嵌套 list（如 [ad_ids[idx][1:]]）。
        """
        # 如果你想修复为“排除某些 id”，可以把下面注释解开，展开 exceptions：
        # flat_exceptions = set()
        # for x in exceptions:
        #     if isinstance(x, (list, tuple, set)):
        #         flat_exceptions.update(x)
        #     else:
        #         flat_exceptions.add(x)

        random_numbers = []
        while len(random_numbers) < count:
            num = random.randint(start, end)
            if num not in exceptions:  # 与原实现一致（可能不会真正过滤到具体 id）
                random_numbers.append(num)
        return random_numbers

# -----------------------------
# 测试集
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.gt_data = []
        self.test_data = []
        self.unitid_data, self.lenth_unit_data = self.load_unit()

        # 这里复用 train.txt：用最后一个作为 GT，其余作为历史
        with open(f"{args.dataset_dir}/{args.test_file}", 'r') as f:
            for line in tqdm(f):
                parts = line.strip().split('\t')
                ad_ids = list(map(int, parts[1].split()))
                ad_filter_ids = []
                for ad_id in ad_ids:
                    if ad_id in self.unitid_data:
                        ad_filter_ids.append(ad_id)
                if len(ad_filter_ids) < 2:
                    # 太短的样本跳过，避免空历史
                    continue
                self.test_data.append({'ad_ids': ad_filter_ids[:-1]})
                self.gt_data.append(ad_filter_ids[-1])
        print(f"test data loaded sucessfully ,{len(self.test_data)}")

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        sample = self.test_data[idx]
        ad_ids = sample["ad_ids"]

        ad_embeddings = []
        for ad_id in ad_ids:
            if ad_id in self.unitid_data:
                ad_embeddings.append(self.unitid_data[ad_id]['embedding'])
            else:
                print(f"{ad_id} not in unit_map")

        gt_ad_id = self.gt_data[idx]
        gt_embedding = self.unitid_data[gt_ad_id]["embedding"]
        return ad_embeddings, gt_ad_id, gt_embedding

    def load_unit(self):
        st = time.time()
        print(f"开始加载unit数据，开始时间{st}")
        root_dir = "data/data322235/w_data/1w_unitid_title_emb"  # 按需修改
        file_list = [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files]
        unitid_data = {}
        for f in file_list:
            safe_process_file(f, unitid_data)
        lenth_unit_data = len(unitid_data)
        if 0 not in unitid_data:
            print("check right: ad_id 0 can be pad id")
        print(f"length unitid {lenth_unit_data}")
        print(f"{time.time() - st}")
        return unitid_data, lenth_unit_data

    def collate_fn(self, batch):
        # 元素：(ad_embeddings(list of vec), gt_id(int), gt_embedding(vec))
        ad_embeddings, gt, gt_embeddings = zip(*batch)
        gt_embeddings = torch.tensor(gt_embeddings, dtype=torch.float32)

        max_len = max(len(emb) for emb in ad_embeddings)

        padded_embeddings = []
        pad_mask = []

        for emb in ad_embeddings:
            emb_len = len(emb)
            padding_len = max_len - emb_len

            if padding_len:
                padding_vector = torch.randn(padding_len, self.args.emb_dim, dtype=torch.float32)
                padded_emb = torch.cat([padding_vector, torch.tensor(emb, dtype=torch.float32)], dim=0)
            else:
                padded_emb = torch.tensor(emb, dtype=torch.float32)

            mask = torch.ones(max_len, dtype=torch.float32)
            if padding_len:
                mask[:padding_len] = 0

            padded_embeddings.append(padded_emb)
            pad_mask.append(mask)

        padded_embeddings = torch.stack(padded_embeddings, dim=0)  # [B, L, D]
        pad_mask = torch.stack(pad_mask, dim=0)                    # [B, L]
        gt_data = torch.tensor(gt, dtype=torch.long)               # [B]

        return padded_embeddings, pad_mask, gt_data, gt_embeddings

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    paretn_dir = os.path.dirname(current_dir)

    dataset_dir = os.path.join(paretn_dir, "data")
    train_file = "train.txt"
    test_file = "test.txt"
    ads_info_file = "ad_data"
    maxlen = 10
    emb_dim = 1024

    class Args:
        dataset_dir = dataset_dir
        train_file = train_file
        test_file = test_file
        maxlen = maxlen
        emb_dim = emb_dim
        unitid_file = ads_info_file

    args = Args()
    train_dataset = TrainDataset(args)
    # test_dataset = TestDataset(args)

    print(f"Train dataset length: {len(train_dataset)}")
    # print(f"Test dataset length: {len(test_dataset)}")
    
    # 测试 getitem
    sample = train_dataset[0]
    ad_embeddings, ad_ids = sample
    print(f"Ad IDs: {ad_ids}")
    print(f"Number of Ad Embeddings: {len(ad_embeddings)}")


    # 测试 collate_fn
    batch = train_dataset.collate_fn([train_dataset[0], train_dataset[1]])
    print(batch)

    print(f"Batch shape: {batch[0].shape}, {batch[1].shape}, {batch[2].shape}, {batch[3].shape}")

    # 测试 TestDataset
    # test_sample = test_dataset[0]
    # test_ad_embeddings, gt_ad_id, gt_embedding = test_sample
    # print(f"GT Ad ID: {gt_ad_id}")
    # print(f"GT Embedding Shape: {gt_embedding.shape}")

    # test_batch = test_dataset.collate_fn([test_dataset[0], test_dataset[1]])
    # print(test_batch)
    # print(f"Test Batch shape: {test_batch[0].shape}, {test_batch[1].shape}, {test_batch[2].shape}, {test_batch[3].shape}")

