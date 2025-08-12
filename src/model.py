import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomContrastiveLoss(nn.Module):
    """
    logits, labels: [B, S, D]
    pad_mask: [B, S]  (1=有效, 0=padding)
    ad_idxs:  [B, S]  (同一 ad_id 视为正样)
    """
    def __init__(self):
        super().__init__()
        self.tau = 0.05  # 温度参数，控制相似度分布的平滑程度

    def forward(self, logits, labels, pad_mask, ad_idxs):
        B, S, D = logits.shape

        # flatten
        logits_flat = logits.reshape(B * S, D)
        labels_flat = labels.reshape(B * S, D)
        valid_flat = pad_mask.reshape(B * S).to(dtype=torch.bool)
        ad_flat = ad_idxs.reshape(B * S)

        # L2 normalize
        logits_flat = F.normalize(logits_flat, p=2, dim=-1)
        labels_flat = F.normalize(labels_flat, p=2, dim=-1)

        # 相似度矩阵 [BS, BS]
        sim = torch.matmul(logits_flat, labels_flat.t())
        sim = sim / self.tau

        # mask：只保留(valid, valid)的行列
        # 形状 [BS, BS]；True=有效，False=无效
        pair_valid = valid_flat.unsqueeze(0) & valid_flat.unsqueeze(1)
        # 将无效位置置为极小，避免 softmax 把概率分给无效列
        sim = sim.masked_fill(~pair_valid, float("-inf"))

        # 正样掩码：同 ad_id 且 (valid, valid)
        pos_mask = (ad_flat.unsqueeze(0) == ad_flat.unsqueeze(1)) & pair_valid

        # 多正样 InfoNCE：
        # loss_i = -log( sum_{j in P_i} exp(sim_ij) / sum_{k in A_i} exp(sim_ik) )
        # 其中 A_i 是该行有效列集合（mask 后）
        pos_sim = sim.masked_fill(~pos_mask, float("-inf"))   # 非正样置 -inf
        pos_lse = torch.logsumexp(pos_sim, dim=-1)            # log sum exp over positives
        all_lse = torch.logsumexp(sim,     dim=-1)            # log sum exp over all valid

        # 仅统计“该行至少有一个正样”的样本
        has_pos = pos_mask.any(dim=-1)  # [BS]
        loss_per_row = -(pos_lse - all_lse)                   # [BS]
        loss = loss_per_row[has_pos].mean()
        return loss
        # # 兜底：行是否全被禁
        # row_all_masked = ~pair_valid.any(dim=-1)         # [BS]
        # if row_all_masked.any():
        #     sim[row_all_masked] = 0.0                    # softmax(0...0)=均匀分布

        # # softmax over columns
        # # sim[i][j]表示预测的第i个位置的embedding与真实的第j个位置的embedding的相似度
        # sim_prob = F.softmax(sim, dim=-1)

        # # 监督：同 ad_id 视为正样（按列）
        # label_mat = (ad_flat.unsqueeze(0) == ad_flat.unsqueeze(1)) & pair_valid
        # label_mat = label_mat.to(sim_prob.dtype)

        # # 只对正样位置计算 -log2(p)
        # # 避免 log(0)
        # eps = 1e-12
        # # 将一个batch内所有非labels的样本全部作为负样本。
        # pos_loss = -torch.log2(sim_prob.clamp_min(eps)) * label_mat

        # # 按行求和，再对有效行取平均
        # loss_per_row = pos_loss.sum(dim=-1)

        # # 仅统计有至少一个正样目标的有效行；若想与原逻辑完全一致，也可直接 mean()
        # # 这里与 Paddle 原写法一致：直接对所有行取均值
        loss = loss_per_row.mean()
        return loss


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B, S, D] -> Conv1d 需要 [B, C, L] 即 [B, D, S]
        y = x.transpose(1, 2)
        y = self.conv1(y)
        y = self.dropout1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.dropout2(y)
        y = y.transpose(1, 2)  # 回到 [B, S, D]
        return x + y  # 残差


class SASRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dev = args.device

        # padding_idx=0 指定索引为0的emb不会参与训练
        self.pos_emb = nn.Embedding(num_embeddings=args.maxlen + 1,
                                    embedding_dim=args.hidden_units,
                                    padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        # hidden维度做 LayerNorm
        self.last_layernorm = nn.LayerNorm(normalized_shape=args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            # 使用 batch_first=True 以匹配 [B, S, D]，指定第一个维度是B
            self.attention_layers.append(
                nn.MultiheadAttention(embed_dim=args.hidden_units,
                                      num_heads=args.num_heads,
                                      dropout=args.dropout_rate,
                                      batch_first=True)
            )
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def log2feats(self, seqs, mask):
        """
        seqs: [B, S, D]  (这里假设已是item embedding或上一层输出)
        mask: [B, S]    (float或bool；1=有效，0=padding)
        """
        B, S, D = seqs.shape
        device = seqs.device

        # # 绝对位置编码（从1开始；padding位置为0 -> 使用 padding_idx=0）
        # pos = torch.arange(1, S + 1, device=device).unsqueeze(0).expand(B, S)
        # pos = pos * mask.to(pos.dtype)  # padding 处为0
        # pos_emb = self.pos_emb(pos.long())  # [B, S, D]
        # mask: True=有效，False=pad
        pos_idx = mask.cumsum(dim=1) * mask      # 左侧 pad 保持 0，右侧得到 1..L
        pos_emb = self.pos_emb(pos_idx.long())

        seqs = seqs + pos_emb
        seqs = self.emb_dropout(seqs)

        # === 构造 3D 组合 mask: [B,S,S], True=屏蔽 ===
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)  # [S,S]
        attn_mask = causal.unsqueeze(0).expand(B, -1, -1).clone()                           # [B,S,S]
        # 屏蔽 key 为 padding 的列
        attn_mask |= (~mask).unsqueeze(1).expand(B, S, S)                                   # [B,S,S]
        # 对 query 为 padding 的行，整行不屏蔽（否则会全 True→-inf）
        attn_mask &= mask.unsqueeze(-1).expand(B, S, S)                                     # [B,S,S]
        # MultiheadAttention 支持 3D attn_mask 形状 [B*h, S, S]
        num_heads = self.attention_layers[0].num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)                           # [B*h,S,S]

        # 也把 Q/K/V 的 pad 行/列置0（更干净）
        pad_row = (~mask).unsqueeze(-1)   # [B,S,1]
        seqs = seqs.masked_fill(pad_row, 0.0)

        for ln_attn, attn, ln_ffn, ffn in zip(
            self.attention_layernorms,
            self.attention_layers,
            self.forward_layernorms,
            self.forward_layers
        ):
            # Pre-LN
            Q = ln_attn(seqs)
            # MultiheadAttention: (B, S, D), (B, S, D), (B, S, D)
            # attn_mask=True 表示 masked
            # mha_out, _ = attn(Q, seqs, seqs, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            mha_out, _ = attn(Q, seqs, seqs, attn_mask=attn_mask)
            # mha_out = mha_out.masked_fill(~mask.unsqueeze(-1), 0.0)  # padding位置置0
            seqs = seqs + mha_out  # 残差

            Y = ln_ffn(seqs)
            seqs = ffn(Y)  # 内部已做残差

        log_feats = self.last_layernorm(seqs)  # [B, S, D]
        return log_feats

    def forward(self, seqs, mask, pos_seqs=None, neg_seqs=None):
        # 训练时返回时序特征 [B, S, D]
        logits = self.log2feats(seqs, mask)
        return logits

    @torch.no_grad()
    def predict(self, seqs, mask, item_embs):
        """
        seqs: [B, S, D]
        mask: [B, S]
        item_embs: [N_items, D]
        return: [B, N_items]
        """
        log_feats = self.log2feats(seqs, mask)
        final_feat = log_feats[:, -1, :]  # 取最后时刻
        logits = torch.matmul(final_feat, item_embs.t())
        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = type('Args', (object,), {
        'maxlen': 200,
        'hidden_units': 1024,
        'num_blocks': 2,
        'num_heads': 1,
        'dropout_rate': 0.2,
        'device': device,
    })()

    # 测试模型结构
    model = SASRec(args).to(device)
    print(model)

    padded_embeddings = torch.randn(32, 200, 1024).to(device)  # 示例输入
    pad_mask = torch.ones(32, 200).to(device)  # 全部有效

    padded_neg_emb = torch.randn(32, 200, 1024).to(device)  # 示例负样本输入
    padded_pos_emb = torch.randn(32, 200, 1024).to(device)  # 示例正样本输入

    logits = model(padded_embeddings, pad_mask, padded_pos_emb, padded_neg_emb)
    print(logits.shape)