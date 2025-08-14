python /root/autodl-tmp/code_pt/build_vector_store.py \
  --ad_file /root/autodl-tmp/code_pt/data/ad_data \
  --out_index ./faiss_idx/ads_flat_ip.faiss \
  --mode flat \
  --batch_size 200000 \
  --normalize \
  --idmap_out ./faiss_idx/ad_id_map.pkl
