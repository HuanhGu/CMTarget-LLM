from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os

from model.multi_fusion import SelfAttention
from model.moe import *
from utils.metrix import *
from embedding.dataset import BertEmbeddings
from model.scorer import Scorer, cross_attention



class CMTargetModel(nn.Module):
    def __init__(self, configs):
        super(CMTargetModel, self).__init__()
        self.configs = configs
        self.stamp = configs['timestamp']
        self.device = configs['device']
        
        self.hidden_dim = configs['hidden_dim']
        self.moe_emb_dim = 1024      # 3 专家编码参数
        self.use_selfatt = configs['use_selfatt']
        self.use_moe = configs['use_moe']
        self.use_cross_att = configs['use_cross_att']

        # 4. 模型可学习参数
        self.protein_embed = BertEmbeddings(vocab_size=30, hidden_size=self.hidden_dim, max_position_embeddings=506, padding_idx=0)
        self.drug_embed = BertEmbeddings(vocab_size=767, hidden_size=self.hidden_dim, max_position_embeddings=222, padding_idx=1)

        # === 创建 fusion 模型 =====
        if self.use_selfatt:
            self.sequence_attention_pro = SelfAttention(self.hidden_dim, self.hidden_dim, self.hidden_dim)
            self.sequence_attention_drug = SelfAttention(self.hidden_dim, self.hidden_dim, self.hidden_dim)
        else:
            print("\nwithout self-attention!!")

        # === 创建 基础专家 模型 ===
        if self.use_moe:
            self.basic_pro_moe = Qwen2MoeSparseMoeBlock(self.hidden_dim, self.moe_emb_dim, 6)
            self.basic_drug_moe = Qwen2MoeSparseMoeBlock(self.hidden_dim, self.moe_emb_dim, 6) 
        else:
            print("without MOE!!")

        # === 创建 打分 模型 ===
        if self.use_cross_att:
            self.cross_attention = cross_attention(head = 8, conv=self.hidden_dim)
        self.scorer = Scorer(configs, self.hidden_dim)


    def forward(self, protein_features, drug_features):
        proteins = protein_features.to(self.device) # 128,447
        drugs = drug_features.to(self.device) # 128,512
        # 掩码
        protein_mask = (proteins != 0).float().unsqueeze(1) #128,1,447
        drug_mask = (drugs != 1).float().unsqueeze(1) # 128,1,512

        # 1. wordEmbedding + PosEmbedding
        proteinembed = self.protein_embed(proteins) #128,447,512
        drugembed = self.drug_embed(drugs) #128,512,768
        pro_hidden = proteinembed
        drug_hidden = drugembed
        
        # 2. 自注意力
        if self.use_selfatt:
            pro_hidden = self.sequence_attention_pro(proteinembed, protein_mask) #128,447,512        
            drug_hidden = self.sequence_attention_drug(drugembed, drug_mask) #128,512,512
        
        contrastive_Loss = 0  # 不要三模态


        # 3. 专家编码器 : 不同蛋白和化合物的token用不同专家编码 
        # 专家编码输出, moe的负载均衡损失
        if self.use_moe:
            pro_hidden, pro_moe_loss = self.basic_pro_moe(pro_hidden, protein_mask) #in:[2,501,100] out:[2,501,256]
            drug_hidden, drug_moe_loss = self.basic_drug_moe(drug_hidden, drug_mask) #in:[2,68,78] out:[2,68,256]
            load_balancing_loss = pro_moe_loss + drug_moe_loss     # 3.9189+3.4380 
        else:
            load_balancing_loss=0


        # 5. 预测最终得分 : 预测蛋白质和化合物的相互作用关系
        if self.use_cross_att:
            drug_hidden ,pro_hidden = self.cross_attention(drug_hidden, pro_hidden)
        else:
            print("without cross-attention!!")

        predicted_scores = self.scorer(pro_hidden, drug_hidden) # 池化+打分
        
        return predicted_scores, contrastive_Loss, load_balancing_loss
    
    
    def save_model(self, output_path = './checkpoints/tmp.pt'):
        # model_path = os.path.join("checkpoints", f"{self.stamp}_{'AttFusion'}.pt")
        torch.save(self.state_dict(), output_path) # 保存权重参数
        
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))




