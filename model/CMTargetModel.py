from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os



from model.multi_fusion import SelfAttention, EnhancedAttentionBlock
from model.moe import *
from utils.metrix import *
from embedding.FeatureExtract import FeatureExtractor
from model.scorer import Scorer
from einops import rearrange

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
    


class CMTargetModel(nn.Module):
    def __init__(self, configs):
        super(CMTargetModel, self).__init__()
        self.configs = configs
        self.stamp = configs['timestamp']
        self.device = configs['device']
        
        self.pro_token_dim = configs['token_dim_pro'] #每个token的维度  probert=100, w2C=100 
        self.drug_token_dim = configs['token_dim_drug']  # 每个token的维度 chemberta 768
        self.moe_emb_dim = 1024      # 3 专家编码参数
        

        # 4. 模型可学习参数
        self.protein_embed = BertEmbeddings(vocab_size=30, hidden_size=self.pro_token_dim, max_position_embeddings=506, padding_idx=0)
        self.drug_embed = BertEmbeddings(vocab_size=767, hidden_size=self.drug_token_dim, max_position_embeddings=222, padding_idx=1)

        # === 创建 fusion 模型 =====
        # self.sequence_attention_pro = EnhancedAttentionBlock(self.pro_token_dim, dropout_rate=0.1)
        # self.sequence_attention_drug = EnhancedAttentionBlock(self.drug_token_dim, dropout_rate=0.1)
        self.sequence_attention_pro = SelfAttention(self.pro_token_dim, 512, 512)
        self.sequence_attention_drug = SelfAttention(self.drug_token_dim, 512, 512)

        # === 创建 基础专家 模型 ===
        # self.basic_pro_moe = BasicMOE(self.pro_token_dim, self.moe_emb_dim, 3)   # (feature_in, feature_out, expert_num)[100,256]
        # self.basic_drug_moe = BasicMOE(self.drug_token_dim, self.moe_emb_dim, 3)   # (feature_in, feature_out, expert_num)[768,256]
        self.basic_pro_moe = Qwen2MoeSparseMoeBlock(512, self.moe_emb_dim, 6)
        self.basic_drug_moe = Qwen2MoeSparseMoeBlock(512, self.moe_emb_dim, 6)   # (feature_in, feature_out, expert_num)[768,256]

        
        # === 创建 打分 模型 ===
        self.scorer = Scorer(configs, self.moe_emb_dim)


    def forward(self, protein_features, drug_features):
        proteins = protein_features.to(self.device)#16,633,100 #128,506
        drugs = drug_features.to(self.device)#16,222,768  # 128,222
        # 掩码
        protein_mask = (proteins != 0).float().unsqueeze(1) #128,1,1200
        drug_mask = (drugs != 1).float().unsqueeze(1) # 128,1,100

        # 1. wordEmbedding + PosEmbedding
        proteinembed = self.protein_embed(proteins) #128,1200   128,1200,768  #128,506,100
        drugembed = self.drug_embed(drugs) #128,222   128,222,128

        # 将embedding通过线性层处理
        # protein_encoder_learn = self.emb_data_pro(proteinembed) # 16,633,100
        # drug_encoder_learn = self.emb_data_drug(drugembed) # 16,222,768

        # 2. 自注意力
        pro_fused_output = self.sequence_attention_pro(proteinembed, protein_mask) #128,633,100        
        drug_fused_output = self.sequence_attention_drug(drugembed, drug_mask)
        contrastive_Loss = 0  # 不要三模态


        # 3. 专家编码器 : 不同蛋白和化合物的token用不同专家编码 
        # 专家编码输出, moe的负载均衡损失
        pro_moe_output, pro_moe_loss = self.basic_pro_moe(pro_fused_output, protein_mask) #in:[2,501,100] out:[2,501,256]
        drug_moe_output, drug_moe_loss = self.basic_drug_moe(drug_fused_output, drug_mask) #in:[2,68,78] out:[2,68,256]
        load_balancing_loss = pro_moe_loss + drug_moe_loss     # 275 + 128    
        load_balancing_loss=0
        # 5. 预测最终得分 : 预测蛋白质和化合物的相互作用关系 in:[2,501,256]  [2,68,256]
        score = self.scorer.forward(pro_moe_output, drug_moe_output)

        return score, contrastive_Loss, load_balancing_loss
    
    
    def save_model(self, output_path = './checkpoints/tmp.pt'):
        # model_path = os.path.join("checkpoints", f"{self.stamp}_{'AttFusion'}.pt")
        torch.save(self.state_dict(), output_path) # 保存权重参数
        
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))




