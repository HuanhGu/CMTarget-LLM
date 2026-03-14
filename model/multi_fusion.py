import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
# import numpy as np


# 实现Scaled Dot-Product Attention
''' S

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    计算Scaled Dot-Product Attention
    
    参数:
        Q: Query矩阵 (batch_size, n_queries, d_k) 或 (n_queries, d_k)
        K: Key矩阵 (batch_size, n_keys, d_k) 或 (n_keys, d_k)
        V: Value矩阵 (batch_size, n_keys, d_v) 或 (n_keys, d_v)
        mask:S 可选的掩码矩阵
    
    返回:
        output: 注意力输出
        attention_weights: 注意力权重矩阵
    """
    d_k = Q.shape[-1]
    
    # Step 1: 计算相似度分数 QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1) if Q.ndimension() == 2 else K.transpose(1, 2))
    
    # Step 2: 缩放
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype, device=scores.device))
    
    # Step 3: 应用掩码（如果有）
    if mask is not None:
        scores = torch.where(mask == 0, torch.tensor(-1e9, dtype=scores.dtype, device=scores.device), scores)
    
    # Step 4: Softmax归一化
    attention_weights = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True).values)
    attention_weights = attention_weights / torch.sum(attention_weights, dim=-1, keepdim=True)
    
    # Step 5: 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
'''

# 定义自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SelfAttention, self).__init__()
        """
        初始化selfAttention

        d_model : 输入维度
        d_k : Query / Key 的维度
        d_v : Value 的维度

        """
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # 初始化权重矩阵（使用Xavier初始化）
        # scale = np.sqrt(2.0 / (d_model + d_k))
        # self.W_Q = np.random.randn(d_model, d_k) * scale
        # self.W_K = np.random.randn(d_model, d_k) * scale
        # self.W_V = np.random.randn(d_model, d_v) * scale
        self.W_Q = nn.Linear(d_model, d_k)  #
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)



    def forward(self, x, mask = None):
        """
        前向传播

        参数：
            x : 输入序列(seq_len, d_model)
            mask : 可选的掩码矩阵

        返回:
            output : 输出序列(seq_len, d_v)
            attention_weights : 注意力权重矩阵

        """
        # 线性投影
        # Q = np.dot(x, self.W_Q) #(seq_len, d_k)
        # K = np.dot(x, self.W_K) #(seq_len, d_k)
        # V = np.dot(x, self.W_V) #(seq_len, d_v)

        Q = self.W_Q(x) #(seq_len, d_k)
        K = self.W_K(x) #(seq_len, d_k)
        V = self.W_V(x) #(seq_len, d_v)

        # 计算注意力
        output= scaled_dot_product_attention(Q, K, V, mask)

        return output




# 对比损失函数（对比学习的核心）
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        """
        对比学习损失( InfoNCE )

        输入 : 
            z_i, z_j : 两种样本表示, 如不同模态, (batch_size, token_num, token_dim)

        参数 : 
            temperature

        输出 : 
            z_i 和 z_j 的对比损失

        """
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # 计算样本级余弦相似度    
        z_i_pool = z_i.mean(dim=1)   # [B, D]
        z_j_pool = z_j.mean(dim=1)   # [B, D]
        sim_matrix = F.cosine_similarity(z_i_pool.unsqueeze(1), z_j_pool.unsqueeze(0), dim=-1) #(batch_size, batch_size)
        # 计算token级余弦相似度  
        # sim_matrix = F.cosine_similarity(
        #     z_i.unsqueeze(2),   # [B, T, 1, D]
        #     z_j.unsqueeze(1),   # [B, 1, T, D]
        #     dim=-1
        # )  #[2, 501, 501]


        # 对比损失公式
        labels = torch.arange(z_i.size(0)).to(z_i.device)  # 对角线为正样本
        loss = F.cross_entropy(sim_matrix / self.temperature, labels)
        return loss

# 定义跨模态对齐融合模型
class CrossModalFusionModel(nn.Module):
    def __init__(self, sequence_dim, structure_dim, knowledge_dim, d_model):
        super(CrossModalFusionModel, self).__init__()
        
        # 自注意力层[d_model, d_k, d_v]
        self.sequence_attention = SelfAttention(d_model, d_model, d_model)
        self.structure_attention = SelfAttention(d_model, d_model, d_model)
        self.knowledge_attention = SelfAttention(d_model, d_model, d_model)
        
        # 融合层
        self.fusion_layer = nn.Linear(d_model * 3, d_model)
        
        # 对比损失
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, sequence_input, structure_input, knowledge_input):
        # 自注意力处理每个模态 [2,501,100] * 3  → [2,501,100] * 3
        sequence_output= self.sequence_attention(sequence_input) #in [16,128,768]
        structure_output = self.structure_attention(structure_input)
        knowledge_output = self.knowledge_attention(knowledge_input)
        
        # 对比学习：分别计算序列、结构和知识模态的对比损失
        seq_struct_loss = self.contrastive_loss(sequence_output, structure_output)
        seq_kg_loss = self.contrastive_loss(sequence_output, knowledge_output)
        struct_kg_loss = self.contrastive_loss(structure_output, knowledge_output)


        # 将输出融合
        combined_input = torch.cat((sequence_output, structure_output, knowledge_output), dim=-1)   #[416,300]
        fused_output = self.fusion_layer(combined_input)
        

        # 融合后的输出
        return fused_output, seq_struct_loss + seq_kg_loss + struct_kg_loss


'''
# 示例输入维度
sequence_dim = 128
structure_dim = 256
knowledge_dim = 64

# 创建模型
model = CrossModalFusionModel(sequence_dim, structure_dim, knowledge_dim)

# 示例输入：序列、结构和知识表示
sequence_input = torch.randn(10, sequence_dim)  # 批量大小为10(10,128)
structure_input = torch.randn(10, structure_dim)
knowledge_input = torch.randn(10, knowledge_dim)

# 前向传播
output, total_loss = model(sequence_input, structure_input, knowledge_input)

# 输出结果
print("Fused Output:", output.shape)  # 输出形状应为 (10, sequence_dim)
print("Total Loss:", total_loss.item())  # 打印总损失值


'''