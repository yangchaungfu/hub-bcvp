import torch
import torch.nn as nn
from transformers import BertModel

'''
基于pytorch的自实现的Bert模型
'''


class BertBaseTorch:
    # 将预训练好的整个权重字典输入进来
    def __init__(self, vocab_size, sentence_len, embedding_dim=768):
        self.vocab_size = vocab_size
        self.sentence_len = sentence_len
        self.hidden_size = embedding_dim
        self.num_attention_heads = 12  # mulit-head机制的head数
        self.num_hidden_layers = 2  # transformer的层数

        self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)  # [词表大小, 768]
        self.segment_embedding = nn.Embedding(2, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(512, self.hidden_size, padding_idx=0)
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_size)

        self.self_attention_q_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.self_attention_k_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.self_attention_v_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.self_attention_out_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.self_attention_layer_norm = nn.LayerNorm(self.hidden_size)

        self.feed_forward_im_linear = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.feed_forward_out_linear = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.feed_forward_layer_norm = nn.LayerNorm(self.hidden_size)

        self.poller_out_linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        print("\n1. Embedding Layer ...")
        print("\t1.1. 词转向量：word embedding, segment embedding, position embedding ...")
        # [batch_size, sentence_len] -> [batch_size, sentence_len, 768]
        we = self.word_embedding(x)
        te = self.segment_embedding(torch.zeros_like(x))
        pe = self.position_embedding(torch.LongTensor(x.shape[0] * [list(range(self.sentence_len))]))

        print("\t1.2. 将这三个embedding 的结果相加，再 layer norm ...")
        embeded_x = self.embedding_layer_norm(we + pe + te)

        sequence_output = None
        print(f"\n2. Transformer Layer ...")
        for i in range(self.num_hidden_layers):
            print(f"  transformer 第{i + 1}层 ...")

            print("\t2.1. self attention, 计算 q,k,v 投影矩阵, q=linear(x), k=linear(x), v=linear(x) ...")
            # shape: [batch_size,sentence_len,768] * [768, 768] + [768] -> [batch_size, sentence_len, 768]
            q = self.self_attention_q_linear(embeded_x)
            k = self.self_attention_k_linear(embeded_x)
            v = self.self_attention_v_linear(embeded_x)

            print("\t2.2. self attention, attention = softmax(q * k.T/√(head_size) * v ...")
            # shape: [batch_size, sentence_len, 768] -> [batch_size, sentence_len, 768]
            attention = self.self_attention(q, k, v)

            print("\t2.3. self attention, 输出 attention_x = linear(attention(k,q,v) ) ...")
            # shape: [batch_size, sen_len, 768] * [768, 768] -> [batch_size, sen_len, 768]
            attention_x = self.self_attention_out_linear(attention)

            print("\t2.4. self attention, attention_x = layer_norm(embeded_x + attention_x) ... ")
            attention_x = self.self_attention_layer_norm(embeded_x + attention_x)

            print("\t2.5. feed forward, 计算 gelu( linear(attention_x) )  ... ")
            # shape: [batch_size, sentence_len, 768] * [768, 4*768] + [4*768] -> [batch_size, sentence_len, 4*768]
            feed_forward_x = nn.GELU()(self.feed_forward_im_linear(attention_x))

            print("\t2.6. feed forward, 输出 feed_forward_x = linear(gelu(linear(x) ... ")
            # shape: [batch_size, sentence_len, 4*768] * [4*768, 768] + [768] -> [batch_size, sentence_len, 768]
            feed_forward_x = self.feed_forward_out_linear(feed_forward_x)

            print("\t2.7. feed forward, sequence_output = layer_norm(attention_x + feed_forward_x) ... ")
            sequence_output = self.feed_forward_layer_norm(attention_x + feed_forward_x)

        print(f"\n3. Pooler_output Layer ... ")
        print("\t  计算：poller_output = tanh( linear(sequence_output[0] )  ... ")
        # shape:  [batch_size, sentence_len, 768] -> [batch_size, 768]
        sequence0 = sequence_output[:, 0, :]  # 取每次批次中每句话的每一个token对应的向量
        poller_output = torch.tanh(self.poller_out_linear(sequence0))

        return sequence_output, poller_output

    def self_attention(self, q, k, v):
        """
        Args:
            q,k,v shape: [batch_size, sentence_len, 768]
        Returns:
            返回参的shape: [batch_size, sentence_len, 768]
        """
        batch_size = q.shape[0]

        # 将 q, k, v 拆分为多头，相当于: [batch_size, sentence_len, 768] -> [batch_size, 12, sentence_len, 768/12] ... ")
        head_size = int(self.hidden_size / self.num_attention_heads)  # muliti-head机制中的dk, 即每个Head的列数
        # shape: [batch_size, sen_len, 768] -> [batch_size, sen_len, 12, 768/12] -> [batch_size, 12, sen_len, 768/12]
        q = q.reshape(batch_size, self.sentence_len, self.num_attention_heads, head_size).transpose(1, 2)
        k = k.reshape(batch_size, self.sentence_len, self.num_attention_heads, head_size).transpose(1, 2)
        v = v.reshape(batch_size, self.sentence_len, self.num_attention_heads, head_size).transpose(1, 2)

        # 计算 attention = softmax(q * k.T/√(head_size) * v
        # shape: [batch_size, 12, sen_len, 768/12] * [batch_size, 12, 768/12, sen_len] -> [batch_size, 12, sen_len, sen_len]
        qk = torch.matmul(q, k.transpose(-2, -1))  # q * k.T
        softmax_qk = torch.softmax(qk / torch.sqrt(torch.LongTensor([head_size])), dim=-1)

        # shape: [batch_size, 12, sen_len, sen_len] * [batch_size, 12, sen_len, 768/12] -> [batch_size, 12, sen_len, 768/12]
        attention = torch.matmul(softmax_qk, v)

        # 合并多头: [batch_size, 12, sen_len, 768/12] -> [batch_size, sen_len, 12, 768/12] -> [batch_size, sen_len, 768]
        return attention.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)


x = torch.LongTensor([[2450, 15486, 102, 2110], [2450, 15486, 102, 2110]])  # 假想成4个字的句子

mybert = BertBaseTorch(21128, 4, 768)
my_sequence_output, my_pooler_output = mybert.forward(x)
print("\nmy_sequence_output: \n", my_sequence_output,
      my_sequence_output.shape)  # shape: [batch_size, sentence_len, 768]
print("\nmy_pooler_output: \n", my_pooler_output, my_pooler_output.shape)  # shape: [batch_size, 768]
