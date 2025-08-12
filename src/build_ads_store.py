# build_ad_store.py
import os, json, sqlite3, numpy as np
from tqdm import tqdm

"""
输入：
  data/ad_data           文本：ad_id \t ... \t "v1,v2,...,vD"
输出（写到 args.dataset_dir）：
  ad_store.sqlite        映射表：map(ad_id TEXT PRIMARY KEY, idx INTEGER)
  ad_vectors.fp16.memmap float16 向量矩阵 [N, D]
  ad_meta.json           {"N":N, "D":D}
"""

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# build_ad_store.py (关键修改处)
def build_store(dataset_dir, ad_data_path, pad_id="0"):
    ensure_dir(dataset_dir)

    # 第一次扫，统计 N 和 D（仅统计合法行）
    N = 0; D = None
    with open(ad_data_path) as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 3: 
                continue
            if D is None:
                D = len(parts[2].split(','))
            # 跳过空向量或维度不匹配
            if len(parts[2].split(',')) != D:
                continue
            N += 1

    print(f"count done: N={N}, D={D}")

    # 预分配：多一行给 PAD
    vec_path = os.path.join(dataset_dir, "ad_vectors.fp16.memmap")
    vectors = np.memmap(vec_path, dtype=np.float16, mode="w+", shape=(N + 1, D))
    vectors[0] = 0  # 0号行保留为 padding 向量（全零）

    # sqlite
    db_path = os.path.join(dataset_dir, "ad_store.sqlite")
    if os.path.exists(db_path): os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=OFF;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("CREATE TABLE map(ad_id TEXT PRIMARY KEY, idx INTEGER);")

    # 可选：显式登记 pad_id -> 0
    cur.execute("INSERT INTO map(ad_id, idx) VALUES(?, ?)", (pad_id, 0))
    conn.commit()

    # 第二次扫，真实数据从 idx=1 开始
    i = 1
    with open(ad_data_path) as f:
        buf = []
        for line in tqdm(f, desc="build"):
            parts = line.strip().split('\t')
            if len(parts) < 3: 
                continue
            ad_id, vec_str = parts[0], parts[2]
            # 跳过 pad_id（避免把真实广告写到 0 号位）
            if ad_id == pad_id:
                continue

            vec = np.fromstring(vec_str, sep=',', dtype=np.float16)
            if vec.shape[0] != D:
                continue

            vectors[i] = vec
            buf.append((ad_id, i))
            i += 1

            if len(buf) >= 50_000:
                cur.executemany("INSERT OR REPLACE INTO map(ad_id, idx) VALUES(?,?)", buf)
                conn.commit(); buf.clear()

        if buf:
            cur.executemany("INSERT OR REPLACE INTO map(ad_id, idx) VALUES(?,?)", buf)
            conn.commit()

    conn.execute("CREATE INDEX idx_map_idx ON map(idx);")
    conn.commit(); conn.close()
    vectors.flush()

    meta = {"N": int(i), "D": int(D), "pad_row": 0, "pad_id": pad_id}
    json.dump(meta, open(os.path.join(dataset_dir, "ad_meta.json"), "w"))
    print("store done:", meta, vec_path, db_path)


if __name__ == "__main__":
    # 按需修改路径
    build_store(
            dataset_dir="/root/autodl-tmp/code_pt/data", 
            ad_data_path="/root/autodl-tmp/code_pt/data/ad_data"
        )
