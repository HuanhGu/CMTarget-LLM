import torch
from torch import nn
import torch.nn.functional as F
from model import *


# from data_process import *


class GMF(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.tune_fcn_GMF = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, user_embedding, item_embedding):
        reaction_result = user_embedding * item_embedding  # [batch_size, max_atom_num, emb_dim]
        output = self.tune_fcn_GMF(reaction_result).squeeze(1)
        output = torch.sigmoid(output)
        return output

class MF(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.tune_linear_MF = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, user_embedding, item_embedding):
        reaction_result = user_embedding * item_embedding  # [batch_size, emb_dim][2,256]
        reaction_result = self.tune_linear_MF(reaction_result) # [2,256] →  [2,1]
        # output = torch.sum(reaction_result, dim=1)
        output = torch.sum(reaction_result, dim=-1) # 修改,当batch=1时，dim就是-1[2]
        output = torch.sigmoid(output)
        return output

class Cosine(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

    def forward(self, user_embedding, item_embedding):
        output = torch.cosine_similarity(user_embedding, item_embedding, dim=1)
        output = torch.sigmoid(output)
        return output



class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SelfAttentionPooling, self).__init__()
        self.tune_linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x): # 输入x维度:[token_num, dim]
        scores = torch.tanh(self.tune_linear1(x))    # (416,256) → (416,128)
        scores = self.linear2(scores)           # (416,128) → (416,1)
        scores = scores.squeeze(-1)             # (416,1) → (416)

        attn_weights = F.softmax(scores, dim=-1)   # [416]

        attn_weights = attn_weights.unsqueeze(-1) # (416) → (416, 1)
        pooled = torch.sum(x * attn_weights, dim=-2)    # [256]
        return pooled, attn_weights
    
class Scorer(torch.nn.Module):
    '''
    给蛋白质和化合物特征向量打分

    输入：
        configs: 配置文件
        pro_feat: 蛋白质特征向量[batch_size, token_num, token_dim]
        drug_feat: 化合物特征向量[batch_size, token_num, token_dim]
        
    输出: 
        output:最终得分 [batch_size]
    '''
    def __init__(self, configs):
        super().__init__()

        self.pro_dim = 256   #pro_dim
        self.drug_dim = 256   #drug_dim
        self.emb_dim = 128    #emb_dim

        self.fea_dim = 256  # pro_dim

        self.user_pooling = SelfAttentionPooling(self.pro_dim, self.emb_dim)
        self.item_pooling = SelfAttentionPooling(self.drug_dim, self.emb_dim)


        if configs['score_way'] == 'MF':
            self.score = MF(self.fea_dim)
        elif configs['score_way'] == 'GMF':
            self.score = GMF(self.fea_dim)
        elif configs['score_way'] == 'Cosine':
            self.score = Cosine(self.fea_dim)

    def forward(self, pro_feat, drug_feat):
        "in:[2,501,256]  [2,68,256]"
        "out:[2]"

        # 1. 将输入映射到同一维度
        prot_pool_feature, _ = self.user_pooling(pro_feat)  # [2,256]
        drug_pool_feature, _ = self.item_pooling(drug_feat) #[2,256]
        
        # 2. 预测打分
        output = self.score(prot_pool_feature, drug_pool_feature) #[2]
        
        return output
