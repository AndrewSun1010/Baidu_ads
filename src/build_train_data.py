# preprocess_train_to_idx.py
import os, sqlite3, gzip, orjson, numpy as np
from tqdm import tqdm

"""
输入：
  data/train.txt         每行: user_id \t "ad_id1 ad_id2 ..."
  data/ad_store.sqlite   上一步构建
输出：
  data/train-00000.jsonl.gz ...  每行: {"user_id": "...", "idx": [i1,i2,...]}
"""

def to_idx_files(dataset_dir, train_txt, shard_size=200_000, maxlen=200):
    conn = sqlite3.connect(os.path.join(dataset_dir, "ad_store.sqlite"))
    cur = conn.cursor()

    def lookup(ad_id):
        row = cur.execute("SELECT idx FROM map WHERE ad_id=?", (ad_id,)).fetchone()
        return None if row is None else int(row[0])

    out_id = 0; count = 0
    fout = gzip.open(os.path.join(dataset_dir, f"train-{out_id:05d}.jsonl.gz"), "wb")

    with open(os.path.join(dataset_dir, train_txt)) as f:
        for line in tqdm(f, desc="to_idx"):
            parts = line.strip().split('\t')
            # if len(parts) < 2: continue
            user_id = parts[0]
            ad_ids = parts[1].split()
            idxs = []
            for s in ad_ids:
                i = lookup(s)
                if i is not None:
                    idxs.append(i)
            if len(idxs) <= 1: 
                continue
            # 只保留尾部 maxlen
            idxs = idxs[-maxlen:]
            obj = {"user_id": user_id, "idx": idxs}
            fout.write(orjson.dumps(obj) + b"\n")

            count += 1
            if count % shard_size == 0:
                fout.close()
                out_id += 1
                fout = gzip.open(os.path.join(dataset_dir, f"train-{out_id:05d}.jsonl.gz"), "wb")

    fout.close(); conn.close()
    print("shards written:", out_id+1)

if __name__ == "__main__":
    to_idx_files(
        dataset_dir="/root/autodl-tmp/code_pt/data", 
        train_txt="train.txt", shard_size=200_000, maxlen=200
        )
