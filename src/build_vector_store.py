#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import argparse
import pickle
from typing import Iterator, Tuple, List, Optional, Dict

import numpy as np
from tqdm import tqdm
import faiss


# =========================
# 解析/归一化
# =========================
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def parse_line(line: str) -> Tuple[int, List[int], np.ndarray]:
    """
    输入行格式：
      广告id \t 广告描述token序列(空格分隔) \t 广告vec(逗号分隔float)
    返回：
      (raw_ad_id, tokens(list[int]), vec(float32[dim]))
    """
    parts = line.strip().split('\t')
    if len(parts) != 3:
        raise ValueError(f"Bad line (need 3 columns): {line[:120]}")
    ad_id = int(parts[0])
    # tokens = [int(t) for t in parts[1].split()] if parts[1] else []
    vec = np.fromstring(parts[2], sep=",", dtype=np.float32)
    if vec.size == 0:
        raise ValueError(f"Empty vector: {line[:120]}")
    return ad_id, parts[1], vec


# =========================
# ID 映射：原始ID -> 连续ID(从 start_id 开始)
# =========================
def load_id_map(path: Optional[str]) -> Dict[int, int]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        id_map = pickle.load(f)
    if not isinstance(id_map, dict):
        raise ValueError(f"id_map in {path} is not a dict")
    return id_map


def save_id_map(id_map: Dict[int, int], path: Optional[str]):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[id_map] saved raw_id -> mapped_id to {path} (size={len(id_map)})")


def next_start_from_id_map(id_map: Dict[int, int], start_id: int) -> int:
    """给定已存在的映射，返回下一个可用的连续ID。"""
    return (max(id_map.values()) + 1) if id_map else start_id


# =========================
# 流式读取 + 映射
# =========================
def stream_read_vectors(
    filepath: str,
    batch_size: int = 200_000,
    dim: Optional[int] = None,
    id_map: Optional[Dict[int, int]] = None,
    start_id: int = 1,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    流式读取广告文件，返回 (mapped_ids[int64], vecs[float32]) 批次。
    - id_map: 原始ad_id -> 映射id；若为None则内部创建新映射（从 start_id 开始）
    - 会就地更新 id_map，外部可在建库结束后保存到磁盘
    """
    ids, vecs = [], []
    expected_dim = dim
    bad, total = 0, 0

    if id_map is None:
        id_map = {}
    next_id = next_start_from_id_map(id_map, start_id)

    with io.open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            total += 1
            if not line.strip():
                continue
            try:
                raw_id, _tokens, v = parse_line(line)

                # 维度对齐
                if expected_dim is None:
                    expected_dim = v.shape[0]
                if v.shape[0] != expected_dim:
                    bad += 1
                    continue

                # 原始ID -> 连续ID（从1开始）
                if raw_id not in id_map:
                    id_map[raw_id] = next_id
                    next_id += 1
                mapped_id = id_map[raw_id]

                ids.append(mapped_id)
                vecs.append(v)

                if len(ids) >= batch_size:
                    yield np.asarray(ids, dtype=np.int64), np.vstack(vecs).astype(np.float32, copy=False)
                    ids, vecs = [], []
            except Exception:
                bad += 1
                continue

    if ids:
        yield np.asarray(ids, dtype=np.int64), np.vstack(vecs).astype(np.float32, copy=False)

    if bad > 0:
        print(f"[stream_read_vectors] skipped {bad} bad/invalid lines over {total} lines.")


# =========================
# 索引构建
# =========================
def build_flat_ip_index_from_file(
    filepath: str,
    batch_size: int,
    normalize: bool,
    id_map: Dict[int, int],
    start_id: int,
) -> faiss.Index:
    """
    精确检索：IndexIDMap2(IndexFlatIP)，用映射后的ID（从1开始）
    返回 CPU 索引（建议保存 CPU 索引，启动时再搬GPU）
    """
    # 先窥探第一批（确定维度并初始化索引）
    gen = stream_read_vectors(filepath, batch_size=batch_size, id_map=id_map, start_id=start_id)
    try:
        ids_b, vecs_b = next(gen)
    except StopIteration:
        raise RuntimeError("No valid vectors read from file.")
    d = vecs_b.shape[1]
    if normalize:
        vecs_b = l2_normalize(vecs_b)

    idmap = faiss.IndexIDMap2(faiss.IndexFlatIP(d))
    idmap.add_with_ids(vecs_b, ids_b)

    for ids_b, vecs_b in gen:
        if normalize:
            vecs_b = l2_normalize(vecs_b)
        idmap.add_with_ids(vecs_b, ids_b)

    return idmap


def build_ivfpq_index_from_file(
    filepath: str,
    batch_size: int,
    nlist: int,
    m: int,
    nbits: int,
    normalize: bool,
    train_cap: int,
    id_map: Dict[int, int],
    start_id: int,
) -> faiss.Index:
    """
    近似检索：IndexIDMap2(IVF-PQ) + 连续ID映射
    返回 CPU 索引
    """
    # 第一次遍历：收集训练样本
    train_pool = []
    pool_cnt = 0
    for ids_b, vecs_b in stream_read_vectors(filepath, batch_size=batch_size, id_map=id_map, start_id=start_id):
        if normalize:
            vecs_b = l2_normalize(vecs_b)
        train_pool.append(vecs_b)
        pool_cnt += vecs_b.shape[0]
        if pool_cnt >= train_cap:
            break
    if not train_pool:
        raise RuntimeError("No training vectors collected.")
    train_mat = np.vstack(train_pool).astype(np.float32, copy=False)
    d = train_mat.shape[1]
    if d % m != 0:
        raise ValueError(f"PQ parameter m={m} must divide dimension d={d}.")

    quantizer = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    print(f"[IVFPQ] training with {train_mat.shape[0]} vectors ...")
    ivfpq.train(train_mat)

    # 第二次遍历：真正建库（需同一映射）
    idmap_index = faiss.IndexIDMap2(ivfpq)
    for ids_b, vecs_b in stream_read_vectors(filepath, batch_size=batch_size, id_map=id_map, start_id=start_id):
        if normalize:
            vecs_b = l2_normalize(vecs_b)
        idmap_index.add_with_ids(vecs_b, ids_b)

    return idmap_index


# =========================
# GPU/保存/加载/检索
# =========================
def to_gpu(index_cpu: faiss.Index, device: int = 0) -> faiss.Index:
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, device, index_cpu)


def to_cpu(index_any: faiss.Index) -> faiss.Index:
    if isinstance(index_any, faiss.GpuIndex):
        return faiss.index_gpu_to_cpu(index_any)
    return index_any


def save_index(index_any: faiss.Index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_index = to_cpu(index_any)
    faiss.write_index(cpu_index, path)
    print(f"[save_index] saved to {path}")


def load_index(path: str, device: Optional[int] = None) -> faiss.Index:
    cpu_index = faiss.read_index(path)
    print(f"[load_index] loaded CPU index from {path}")
    if device is not None:
        gpu_index = to_gpu(cpu_index, device=device)
        print(f"[load_index] moved to GPU:{device}")
        return gpu_index
    return cpu_index


def search_topk(
    index: faiss.Index,
    query_vecs: np.ndarray,
    topk: int = 10,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    q = query_vecs.astype(np.float32, copy=False)
    if normalize:
        q = l2_normalize(q)
    topk = min(topk, index.ntotal)
    scores, ids = index.search(q, topk)
    return scores, ids


# =========================
# CLI
# =========================
def main():
    # parser = argparse.ArgumentParser(description="Build FAISS index from ad file with ID remapping (start from 1).")
    # parser.add_argument("--ad_file", type=str, required=True, help="广告文件路径：id\\ttokens\\tvector")
    # parser.add_argument("--out_index", type=str, required=True, help="输出索引路径（.faiss）")
    # parser.add_argument("--mode", type=str, default="flat", choices=["flat", "ivfpq"], help="flat=精确；ivfpq=近似")
    # parser.add_argument("--batch_size", type=int, default=200000, help="流式批大小")
    # parser.add_argument("--normalize", action="store_true", help="是否L2归一化（余弦）")
    # parser.add_argument("--gpu", action="store_true", help="保存前是否搬到GPU（通常不需要）")

    # # IVF-PQ 参数
    # parser.add_argument("--nlist", type=int, default=4096, help="IVF簇数")
    # parser.add_argument("--m", type=int, default=32, help="PQ子向量数（需整除维度）")
    # parser.add_argument("--nbits", type=int, default=8, help="PQ每子向量位数")
    # parser.add_argument("--train_cap", type=int, default=200000, help="IVFPQ训练样本上限")

    # # ID 映射
    # parser.add_argument("--idmap_in", type=str, default=None, help="已有 raw->mapped 映射pkl（可复用）")
    # parser.add_argument("--idmap_out", type=str, default="./faiss_idx/ad_id_map.pkl", help="保存 raw->mapped 映射pkl")
    # parser.add_argument("--start_id", type=int, default=1, help="连续ID的起始（默认1）")

    # args = parser.parse_args()

    idmap_in = None
    start_id = 1
    mode = 'flat'
    ad_file = '/root/autodl-tmp/code_pt/data/ad_data'
    batch_size = 200000
    normalize = True
    out_index = './faiss_idx/ads_flat_ip.faiss'
    idmap_out = './faiss_idx/ad_id_map.pkl'
    gpu = False
    # 加载/准备映射
    id_map = load_id_map(idmap_in)
    if id_map:
        print(f"[id_map] loaded {len(id_map)} entries from {idmap_in}")
    else:
        print("[id_map] start fresh mapping from", start_id)

    # 建库
    if mode == "flat":
        index_cpu = build_flat_ip_index_from_file(
            filepath=ad_file,
            batch_size=batch_size,
            normalize=normalize,
            id_map=id_map,
            start_id=start_id,
        )
    else:
        # index_cpu = build_ivfpq_index_from_file(
        #     filepath=args.ad_file,
        #     batch_size=args.batch_size,
        #     nlist=args.nlist,
        #     m=args.m,
        #     nbits=args.nbits,
        #     normalize=args.normalize,
        #     train_cap=args.train_cap,
        #     id_map=id_map,
        #     start_id=args.start_id,
        # )
        pass

    # 保存索引
    index_to_save = to_gpu(index_cpu) if gpu else index_cpu
    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    save_index(index_to_save, out_index)

    # 保存映射（完成后统一落盘，保证和索引是一致的）
    save_id_map(id_map, idmap_out)


if __name__ == "__main__":
    main()
