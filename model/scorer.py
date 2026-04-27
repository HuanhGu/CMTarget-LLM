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
        # output = torch.sigmoid(output)
        return output

class MF(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # self.emb_dim = emb_dim
        # self.tune_linear_MF = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, user_embedding, item_embedding):
        # reaction_result = user_embedding * item_embedding  # [batch_size, emb_dim][2,256]
        # reaction_result = self.tune_linear_MF(reaction_result) # [2,256] →  [2,1]
        # # output = torch.sum(reaction_result, dim=1)
        # output = output.squeeze(-1) # 去掉多余的 1 维度，变成 [batch_size]
        # output = torch.sigmoid(output)
        # 直接计算点积并求和: [batch_size]
        output = torch.sum(user_embedding * item_embedding, dim=-1)
        # output = torch.sigmoid(output)
        return output

class Cosine(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # self.temp = nn.Parameter(torch.tensor([10.0]))

    def forward(self, user_embedding, item_embedding):
        output = torch.cosine_similarity(user_embedding, item_embedding, dim=1)
        # output = torch.sigmoid(output)
        # output = cos * self.temp
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
        '''
        attn_weights = F.softmax(scores, dim=-1)   # [416]
        attn_weights = attn_weights.unsqueeze(-1) # (416) → (416, 1)
        pooled = torch.sum(x * attn_weights, dim=-2)    # [256]
        '''

        attn_weights = torch.sigmoid(scores) # (416) 每个 token 独立打分
        attn_weights = attn_weights.unsqueeze(-1)
        pooled = torch.sum(x * attn_weights, dim=-2) / (torch.sum(attn_weights, dim=-2) + 1e-6)
        return pooled, attn_weights

class mutil_head_attention(nn.Module):
    def __init__(self,head = 8,conv=32):
        super(mutil_head_attention,self).__init__()
        self.conv = conv #256
        self.head = head
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.d_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        # self.p_a = nn.Linear(self.conv * 3, self.conv * 3 * head)
        # self.scale = torch.sqrt(torch.FloatTensor([self.conv * 3])).cuda()
        self.d_a = nn.Linear(self.conv, self.conv * self.head)
        self.p_a = nn.Linear(self.conv, self.conv* self.head)
        self.scale = torch.sqrt(torch.FloatTensor([self.conv])).cuda()

    def forward(self, drug, protein):
        bsz, d_il, d_ef = drug.shape  #128,633,256
        bsz, p_il, p_ef = protein.shape  #12,222,256
        drug_att = self.relu(self.d_a(drug)).view(bsz,self.head,d_il,d_ef) #128,8,633,256
        protein_att = self.relu(self.p_a(protein)).view(bsz,self.head,p_il,p_ef)#123,8,222,256
        interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1) #128,633,222
        Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(2) #128,85,1179 → 128,1,85
        Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(2)    #128,85,1179 → 128,1,1179
        drug = drug * Compound_atte  #128,96,85
        protein = protein * Protein_atte #128,96,1179
        return drug,protein #128,633,256;  128,222,256
    
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
    def __init__(self, configs, moe_emb_dim):
        super().__init__()
        self.fea_dim = moe_emb_dim  # pro_dim, drug_dim, 256
        self.emb_dim = self.fea_dim

        self.attention = mutil_head_attention(head = 8, conv=self.fea_dim)
        # self.Drug_max_pool = nn.AdaptiveMaxPool1d(1)
        # self.Protein_max_pool = nn.AdaptiveMaxPool1d(1)
        # self.Drug_avg_pool = nn.AdaptiveAvgPool1d(1)
        # self.Protein_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.user_pooling = SelfAttentionPooling(self.fea_dim, self.emb_dim)
        self.item_pooling = SelfAttentionPooling(self.fea_dim, self.emb_dim)
        

        if configs['score_way'] == 'MF':
            self.score = MF(self.fea_dim)
        elif configs['score_way'] == 'GMF':
            self.score = GMF(self.fea_dim)
        elif configs['score_way'] == 'Cosine':
            self.score = Cosine(self.fea_dim)
        # elif configs['score_way'] == 'MLP':#[直接排除]
        #     self.score = nn.Sequential(
        #         nn.Linear(self.fea_dim*4, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(1024, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 1) # 输出打分
        #     )
        # elif configs['score_way'] == 'bilinear':
        #      # score = x^T * W * y
        #     self.score = nn.Bilinear(in1_features=self.fea_dim, in2_features=self.fea_dim, out_features=1)

        

    def forward(self, pro_feat, drug_feat):
        "in:[128,506,512]  [128,222,768]"
        "out:[2]"

        # 1. 使用多头注意力，将蛋白质和化合物进行特征交互
        pro_feat_mutual ,drug_feat_mutual = self.attention(pro_feat, drug_feat)
        
        """
        # drug_pool_feature = self.Drug_max_pool(drug_feat_mutual.permute(0, 2, 1)).squeeze(2)
        # prot_pool_feature = self.Protein_max_pool(pro_feat_mutual.permute(0, 2, 1)).squeeze(2)
        
        # --- 药物特征池化 ---
        drug_max = self.Drug_max_pool(drug_feat_mutual.permute(0, 2, 1)).squeeze(2)
        drug_avg = self.Drug_avg_pool(drug_feat_mutual.permute(0, 2, 1)).squeeze(2)
        # 拼接后的维度将是原来的 2 倍
        drug_pool_feature = torch.cat([drug_max, drug_avg], dim=1)
        # --- 蛋白质特征池化 ---
        prot_max = self.Protein_max_pool(pro_feat_mutual.permute(0, 2, 1)).squeeze(2)
        prot_avg = self.Protein_avg_pool(pro_feat_mutual.permute(0, 2, 1)).squeeze(2)
        # 拼接后的维度将是原来的 2 倍
        prot_pool_feature = torch.cat([prot_max, prot_avg], dim=1)
        """

        # 1. 将输入映射到同一维度
        prot_pool_feature, _ = self.user_pooling(pro_feat_mutual)  # [2,256]
        drug_pool_feature, _ = self.item_pooling(drug_feat_mutual) #[2,256]
        

        # 2. 预测打分
        output = self.score(prot_pool_feature, drug_pool_feature) #[2]
        
        # MLP
        # combined = torch.cat([drug_pool_feature, prot_pool_feature], dim=1) # 维度: [B, 512]
        # score = self.classifier(combined)
        
        return output

