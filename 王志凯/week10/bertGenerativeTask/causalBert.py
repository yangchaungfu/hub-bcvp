# -*- coding:utf-8 -*-

import torch
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import BertModel
from logHandler import logger
log = logger(__file__)


class CausalBertSelfAttention(BertSelfAttention):
    """
    # BERT中一个注意力层的简化结构:
    BertLayer
    ├── BertAttention
    │   ├── BertSelfAttention    <-  要修改这里！
    │   │   ├── query (线性层)
    │   │   ├── key (线性层)
    │   │   └── value (线性层)
    │   └── BertSelfOutput (输出投影)
    ├── BertIntermediate (前馈网络)
    └── BertOutput (前馈输出)
    """

    # 重写BertSelfAttention中的forward方法（需要对参数attention_mask重新赋值，改为下三角掩码）
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, **kwargs):

        # 当encoder_hidden_states不为空时，表示当前为交叉注意力层，即decoder的第二层
        is_self_attention = encoder_hidden_states is None

        # 当encoder_hidden_states为空时，表示是encoder的自注意力层，正是需要修改的地方
        if is_self_attention:
            # hidden_states为上一个网络层传递过来的数据（embedding层的输出）
            batch_size, seq_len, char_dim = hidden_states.shape
            log.info(f"hidden_states.shape:{hidden_states.shape}")

            # 创建下三角掩码
            """
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
            """
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=hidden_states.device))
            # 在原始的BertSelfAttention中，会使用注意力得分加上掩码（score + mask_attention）
            # 所以这里需要将1置为0，将0置为无穷小，最后softmax(score + mask_attention)就相当于将未来词概率置为0
            causal_mask = (1 - causal_mask) * - 1e9     # (seq_len, seq_len)
            # 因为score是一个四维张量（batch_size, head, seq_len, seq_len）,所以causal_mask必须也为四维才能相加
            causal_mask.unsqueeze(0).unsqueeze(1)       # (1, 1, seq_len, seq_len)

            # 还要处理原始的 attention_mask
            # 它可能是encoder或者decoder部分传来的2D（batch_size, seq_len）的掩码，也可能是3D(batch_size, seq_len, seq_len)或4D
            # 需要和causal_mask维度一致才能合并
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                else:
                    pass
                # 合并掩码
                attention_mask = attention_mask + causal_mask
            else:
                attention_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)

        # 重写完成，直接调用父类forward
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs)


def causalBertModel(config):
    model = BertModel.from_pretrained(config["bert_path"], return_dict=True)
    # 修改每一层的注意力
    for layer in model.encoder.layer:
        # 加载改写后的attention
        causal_attention = CausalBertSelfAttention(model.config)
        # 加载权重
        causal_attention.load_state_dict(layer.attention.self.state_dict())
        # 置换
        layer.attention.self = causal_attention
    return model