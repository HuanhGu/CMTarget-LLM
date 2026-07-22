import torch
from torch import nn
from torch.utils.data import Dataset

from embedding.FeatureExtract import FeatureExtractor
from einops import rearrange

class DTIDataset(Dataset):
    '''
    输入[df] : 包含 (compound, protein, label) 的 pandas DataFrame
    '''
    def __init__(self, d_df):
        self.fe = FeatureExtractor()
        # 获取所有化合物序列的最大token数量/平均token数量，便于特征对齐
        self.d_max_tokenLen = self.fe.get_chemberta_max_length(d_df['compound'].tolist()) #222
        self.p_max_tokenLen = self.fe.get_probert_max_length(d_df['protein'].tolist()) #506
        # self.p_max_tokenLen, p_max_kmers = self.feature_extractor.get_protein_max_kmers(d_df['protein'].tolist())

        self.data = d_df.reset_index(drop=True)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        compound = row["compound"]
        protein = row["protein"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        drug_ids = self.fe.drug_tokenizer_chemberta(compound, self.d_max_tokenLen)
        protein_ids = self.fe.pro_tokenizer_probert(protein, self.p_max_tokenLen)

        drug_ids = torch.as_tensor(drug_ids).squeeze()
        protein_ids = torch.as_tensor(protein_ids).squeeze()
        return drug_ids, protein_ids, label
    


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # γ

    def forward(self, x):
        # x: [batch, seq_len, dim]
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()  # 计算 RMS
        x_normed = x / (rms + self.eps)               # 归一化
        return x_normed * self.scale

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=767, hidden_size=768, max_position_embeddings=514, padding_idx=1):
        super(BertEmbeddings, self).__init__()
        
        # 1. 词向量层: 词典大小 767, 维度 768
        # padding_idx=1 表示索引为 1 的 token (通常是 <pad>) 不计入梯度更新
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.token_type_embeddings = nn.Embedding(1, hidden_size)# 3. 句子类型向量层

        # Rotary Positional Embedding
        self.rope = RotaryPositionEmbedding(hidden_size)

        # self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-05)
        self.LayerNorm = RMSNorm(hidden_size, eps=1e-05)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, input_ids, token_type_ids=None):
        # 获取序列长度
        seq_length = input_ids.size(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 如果没有传入 token_type_ids，默认为全 0
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 词向量 + token type
        words_emb = self.word_embeddings(input_ids)
        type_emb = self.token_type_embeddings(token_type_ids)
        embeddings = words_emb + type_emb

        # RoPE 位置编码
        pos_emb = self.rope(seq_length, device)
        embeddings = apply_rotary_pos_emb(pos_emb, embeddings)

        # 最后进行归一化和 Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    