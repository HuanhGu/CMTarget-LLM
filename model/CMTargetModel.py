from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os


# from feature_extract import *
# from interaction_pred import *
# from scorer import *

# 这个路径导入要看运行的当前文件夹, 不是当前文件的文件夹
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *
from embedding.FeatureExtract import FeatureExtractor

from model.scorer import Scorer


class CMTargetModel(nn.Module):
    def __init__(self, configs):
        super(CMTargetModel, self).__init__()
        self.configs = configs
        self.stamp = configs['timestamp']
        # self.feature_extractor = feature_extractor
        self.device = configs['device']

        #  2 特征融合的超参数
        self.pro_sequence_tklen = 416         # 蛋白质序列编码的token数目
        self.pro_structure_tklen = 416 # 256  # 蛋白质结构编码的token数目
        self.pro_knowledge_tklen = 416 # 64   # 蛋白质知识图谱提取的token数目 
        self.pro_token_dim = 100              # 每个token的维度 100


        self.drug_sequence_tklen = 43          # 化合物序列编码的token数目
        self.drug_structure_tklen = 43         # 化合物结构编码的token数目
        self.drug_knowledge_tklen = 43         # 化合物知识图谱提取的token数目 
        self.drug_token_dim = 768              # 每个token的维度 768


        # 3 专家编码超参数
        self.pro_fusion_dim = 100
        self.drug_fusion_dim = 768
        self.pro_moe_dim = 256
        self.drug_moe_dim = 256

        
        # 5. 预测超参数
        self.score_way = configs['score_way']   # 打分器选择
        self.score_emb_dim = 128                # 打分时的特征嵌入维度
        
        # 6. 模型可学习参数
        # === 创建 fusion 模型 =====
        self.pro_fusion_model = CrossModalFusionModel(self.pro_sequence_tklen, self.pro_structure_tklen, self.pro_knowledge_tklen, self.pro_token_dim)
        self.drug_fusion_model = CrossModalFusionModel(self.drug_sequence_tklen, self.drug_structure_tklen, self.drug_knowledge_tklen, self.drug_token_dim)
        

        # === 创建 基础专家 模型 ===
        self.basic_pro_moe = BasicMOE(self.pro_fusion_dim, self.pro_moe_dim, 3)   # (feature_in, feature_out, expert_num)[100,256]
        self.basic_drug_moe = BasicMOE(self.drug_fusion_dim, self.drug_moe_dim, 3)   # (feature_in, feature_out, expert_num)[768,256]
        
        # === 创建 打分 模型 ===
        # self.predictor = Scorer(self.pro_moe_dim, self.drug_moe_dim, self.score_emb_dim, "MF")
        self.scorer = Scorer(configs)


    def forward(self, protein_features, drug_features):
        """
        model的前向传播

        输入:
            drug_features:[batch_size,] 化合物特征向量
            protein_features : [batch_size, ] 蛋白质特征向量
            // protein_batch: [batch_size, ]蛋白质序列
            // compound_batch: [batch_size, ]化合物序列
            // pro_encoder_modals: [3, batch_size, token_num, token_dim] 蛋白质三种模态的特征encoder tensor
            // drug_encoder_modals:[3, batch_size, token_num, token_dim] 化合物三种模态的特征encoder tensor
            

        返回:
            pro_moe_output: 蛋白质序列经过特征提取、融合、moe编码后的特征向量, 
            drug_moe_output:化合物序列经过特征提取、融合、moe编码后的特征向量,   
            pro_emb_loss:   protein的特征嵌入损失
            drug_emb_loss:  drug的特征嵌入损失
        """
        protein_features = protein_features.to(self.device)
        drug_features = drug_features.to(self.device)

        # 构造三模态
        # print(type(protein_features))
        pro_encoder_modals = torch.stack(
            [protein_features, protein_features, protein_features],
            dim=0
        )  # (3, B, T, D)  [3,2,416,100]  [3,2,195,100] [3,16,1024]

        drug_encoder_modals = torch.stack(
            [drug_features, drug_features, drug_features],
            dim=0
        )   # [3,2,43,768]  [3,2,73,768]  
        

        # 2. 特征融合 —— 采用注意力机制
        # 2.1  蛋白质特征融合;2.2  化合物特征融合
        # pro_X = [3,2,501,100]  →  [2,501,100]    drug_X:[3,2,68,768] →  [2,68,768] 
        # 前向传播, 对比损失
        pro_fused_output, pro_fusion_loss = self.pro_fusion_model(pro_encoder_modals[0], pro_encoder_modals[1], pro_encoder_modals[2])
        drug_fused_output, drug_fusion_loss = self.drug_fusion_model(drug_encoder_modals[0], drug_encoder_modals[1], drug_encoder_modals[2])
        contrastive_Loss = pro_fusion_loss + drug_fusion_loss

        # 3. 专家编码器 : 不同蛋白和化合物的token用不同专家编码 
        # 专家编码输出, moe的负载均衡损失
        pro_moe_output, pro_moe_loss = self.basic_pro_moe(pro_fused_output) #in:[2,501,100] out:[2,501,256]
        drug_moe_output, drug_moe_loss = self.basic_drug_moe(drug_fused_output) #in:[2,68,78] out:[2,68,256]
        load_balancing_loss = pro_moe_loss + drug_moe_loss        

        # 5. 预测最终得分 : 预测蛋白质和化合物的相互作用关系 in:[2,501,256]  [2,68,256]
        score = self.scorer.forward(pro_moe_output, drug_moe_output)

        # pro_train_loss = pro_fusion_loss + pro_moe_loss
        # drug_train_loss = drug_fusion_loss + drug_moe_loss
        return score, contrastive_Loss, load_balancing_loss
    
    
    def save_model(self, output_path = './checkpoints/tmp.pt'):
        # model_path = os.path.join("checkpoints", f"{self.stamp}_{'AttFusion'}.pt")
        torch.save(self.state_dict(), output_path) # 保存权重参数
        
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))




