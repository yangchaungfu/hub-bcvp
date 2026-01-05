import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力层"""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model必须能被n_head整除"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head  # 每个头的维度

        # 线性变换层（Q/K/V/输出）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)  # 缩放因子

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        计算缩放点积注意力
        Args:
            q: [batch_size, n_head, seq_len_q, d_k]
            k: [batch_size, n_head, seq_len_k, d_k]
            v: [batch_size, n_head, seq_len_v, d_k]
            mask: [batch_size, 1, seq_len_q, seq_len_k] 或 [batch_size, n_head, seq_len_q, seq_len_k]
        Returns:
            注意力输出和注意力权重
        """
        # 计算注意力分数: (Q @ K^T) / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 应用掩码（填充掩码/未来掩码）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重并dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            q/k/v: [batch_size, seq_len, d_model]
            mask: 注意力掩码
        Returns:
            多头注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)

        # 线性变换并拆分多头: [batch_size, seq_len, d_model] -> [batch_size, n_head, seq_len, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # 拼接多头输出: [batch_size, n_head, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性变换
        output = self.w_o(attn_output)
        return output, attn_weights


class FeedForwardNetwork(nn.Module):
    """前馈网络：两层线性变换 + ReLU + Dropout"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            前馈网络输出 [batch_size, seq_len, d_model]
        """
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """编码器单层：自注意力 + 前馈网络 + 残差连接 + 层归一化"""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # 层归一化和残差连接相关
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        """
        Args:
            x: 编码器输入 [batch_size, src_len, d_model]
            src_mask: 源序列掩码（填充掩码）
        Returns:
            编码器单层输出 [batch_size, src_len, d_model]
        """
        # 自注意力 + 残差连接 + 层归一化（Pre-LN结构）
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class DecoderLayer(nn.Module):
    """解码器单层：掩码自注意力 + 编码器-解码器注意力 + 前馈网络"""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)  # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)  # 编码器-解码器注意力
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # 层归一化和dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor = None,
                src_tgt_mask: torch.Tensor = None):
        """
        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            enc_output: 编码器输出 [batch_size, src_len, d_model]
            tgt_mask: 目标序列掩码（未来掩码+填充掩码）
            src_tgt_mask: 源-目标序列掩码（源填充掩码）
        Returns:
            解码器单层输出 [batch_size, tgt_len, d_model]
        """
        # 1. 掩码自注意力 + 残差 + 层归一化
        attn1_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn1_output))

        # 2. 编码器-解码器注意力 + 残差 + 层归一化
        attn2_output, _ = self.cross_attn(x, enc_output, enc_output, src_tgt_mask)
        x = self.norm2(x + self.dropout2(attn2_output))

        # 3. 前馈网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(
            self,
            src_vocab_size: int,  # 源语言词汇表大小
            tgt_vocab_size: int,  # 目标语言词汇表大小
            d_model: int = 512,  # 模型维度
            n_head: int = 8,  # 注意力头数
            num_encoder_layers: int = 6,  # 编码器层数
            num_decoder_layers: int = 6,  # 解码器层数
            d_ff: int = 2048,  # 前馈网络中间维度
            max_len: int = 5000,  # 最大序列长度
            dropout: float = 0.1  # dropout概率
    ):
        super().__init__()

        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码层
        self.pos_encoding = nn.Embedding(512, d_model)

        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
        """
        生成各类掩码：
        - 源序列填充掩码
        - 目标序列填充掩码
        - 目标序列未来掩码（防止看到未来token）
        """
        batch_size, src_len = src.size()
        batch_size, tgt_len = tgt.size()

        # 源序列填充掩码: [batch_size, 1, 1, src_len]
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

        # 目标序列填充掩码: [batch_size, 1, 1, tgt_len]
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

        # 目标序列未来掩码: [1, 1, tgt_len, tgt_len]
        tgt_future_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).bool()

        # 目标序列最终掩码（填充+未来）: [batch_size, 1, tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_future_mask

        # 源-目标掩码（用于编码器-解码器注意力）: [batch_size, 1, tgt_len, src_len]
        src_tgt_mask = (src != pad_idx).unsqueeze(1).unsqueeze(3)

        return src_mask, tgt_mask, src_tgt_mask

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        """编码器前向传播"""
        # 词嵌入 + 位置编码
        src_emb = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src_emb = self.pos_encoding(src_emb)

        # 编码器层堆叠
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        return enc_output

    def decode(self, tgt: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor, src_tgt_mask: torch.Tensor):
        """解码器前向传播"""
        # 词嵌入 + 位置编码
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt_emb = self.pos_encoding(tgt_emb)

        # 解码器层堆叠
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, src_tgt_mask)
        return dec_output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
        """
        完整前向传播
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            pad_idx: 填充token的索引
        Returns:
            输出logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # 生成掩码
        src_mask, tgt_mask, src_tgt_mask = self.generate_mask(src, tgt, pad_idx)

        # 编码
        enc_output = self.encode(src, src_mask)

        # 解码
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_tgt_mask)

        # 输出层
        output = self.fc_out(dec_output)
        return output
