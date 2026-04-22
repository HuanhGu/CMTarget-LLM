import warnings
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from rdkit.Chem import AllChem
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from rdkit import Chem
from gensim.models import Word2Vec
import torch.nn as nn


class FeatureExtractor(object):
    """
    pre-extract the feature of protein_sequence using word2vec, 
    pre-extract the feature of drug_smiles using chemberta.
    output_dim : [batch_size, token_len, token_feature_dim]
    """
    def __init__(self):
        # self.configs = configs
        self.feature_dim = 1024
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 预加载模型，避免在循环中重复加载
        print("Loading Word2Vec model...")
        self.w2v_model = Word2Vec.load("./embedding/word2vec_30.model")
        
        print("Loading ChemBERTa model...")
        '''
        # 在本地使用, 直接加载缓存模型即可
        self.drug_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", local_files_only=True)
        self.drug_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", local_files_only=True).to(self.device)
        '''

        '''
        # 在服务器上使用, 先下载, 再使用下面这段代码加载模型
        # 把chembert下载到本地
        export HF_ENDPOINT="https://hf-mirror.com"
        huggingface-cli download --resume-download seyonec/ChemBERTa-zinc-base-v1 --local-dir ./embedding/ChemBERTa
        '''
        
        local_model_path = "/root/gpufree-data/workplace/CMTarget-LLM/embedding/ChemBERTa/"
        self.drug_tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        self.drug_model = AutoModel.from_pretrained(local_model_path, local_files_only=True).to(self.device)
        self.drug_model.eval()
    


    def get_protein_embedding(self, model,protein):
        """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    
        vec = np.zeros((len(protein), 100))
        i = 0
        for word in protein:
            vec[i, ] = model.wv[word]
            i += 1
        return vec

    def pro_fea_extract(self, pro_sequence, p_max_kmers):
        '''
        提取一个batch蛋白质序列的特征编码tensor

        输入：
            蛋白质序列list : [batch_size, ]个sequence
        输出：
            蛋白质序列的张量嵌入list : [batch_size, (token_num), Hidden_Size ]
        '''
        proteins = []
        # 假设每个 k-mer 的向量维度是 embedding_dim (比如 100)
        embedding_dim = self.w2v_model.vector_size

        # dummy_tensor = torch.zeros(p_max_kmers)

        for seq in pro_sequence:
            # 将蛋白质序列切分成多个氨基酸
            k = 3
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            
            # 查表操作在 CPU 上完成
            vec_array = np.array([self.w2v_model.wv[w] for w in kmers if w in self.w2v_model.wv])
            vec = torch.FloatTensor(vec_array)

            # 4. 核心填充/截断逻辑
            curr_len = vec.size(0)
            if curr_len < p_max_kmers:
                padded_v = torch.zeros((p_max_kmers, embedding_dim))# 填充：[目标长度, 向量维度]
                padded_v[:curr_len, :] = vec
            else:
                padded_v = vec[:p_max_kmers, :]  # 截断

            proteins.append(padded_v) #[[501,100], [500,100]]

        # 最后直接堆叠成一个 Batch 张量
        proteins_tensor = torch.stack(proteins) 
        # 输出形状: [Batch_Size, p_max_kmers, embedding_dim]

        return proteins_tensor # [8,619,100]
    

    def get_protein_max_kmers(self, proteins):
        "获取一批蛋白质序列划分为氨基酸后的最大长度"
        
        p_kmers = []
        for seq in proteins:
            k = 3
            kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
            p_kmers.append(len(kmers))
        
        p_mean_kmers = int(np.mean(p_kmers))
        p_max_kmers = int(np.max(p_kmers))
        print(f"protein序列 的 全局平均 氨基酸 数量为: {p_mean_kmers}")
        print(f"protein序列 的 全局最大 氨基酸 数量为: {p_max_kmers}")
        return p_mean_kmers, p_max_kmers
    

    def pro_fea_extract_probert(self, pro_sequence, p_max_tokenLen):
        # padding : pad_to_max_length
        inputs = self.pro_tokenizer(pro_sequence, return_tensors="pt", 
                                     padding='max_length', max_length=p_max_tokenLen,
                                     truncation=True).to(self.device) # input_ids : [batch, d_max_tokenLen] || [8, 222]
        # mask = inputs['attention_mask'].cpu()
        with torch.no_grad():
            outputs = self.pro_model(**inputs)
        
        # 结果转回 CPU 释放显存   
        # drugs = outputs.pooler_output [batch, 768]
        proteins = outputs.last_hidden_state.cpu() # [batch, d_max_tokenLen, 78] [8,222,768]  || [8, 72, 768], [8, 83, 768]
        return proteins
    
    def get_probert_max_length(self, all_pro_smiles):
            all_inputs = self.pro_tokenizer(all_pro_smiles, truncation=True)
            # 获取所有编码后的input_ids的长度，取最大值
            max_prolen_all = max([len(x) for x in all_inputs['input_ids']])
            print(f"pro_smiles 的 全局最大 token 长度为: {max_prolen_all}")

            return max_prolen_all

    # https://github.com/miservilla/ChemBERTa
    def drug_fea_extract_chemberta(self, drug_sequence, d_max_tokenLen):
        """
        提取一个batch化合物序列的特征编码tensor

        输入：
            drug序列list : [batch_size, ]个 list of SMILES
        输出：
            drug序列的张量嵌入list : [batch_size, d_max_token_num, Hidden_Size]
        """
        # padding : pad_to_max_length
        inputs = self.drug_tokenizer(drug_sequence, return_tensors="pt", 
                                     padding='max_length', max_length=d_max_tokenLen,
                                     truncation=True).to(self.device) # input_ids : [batch, d_max_tokenLen] || [8, 222]
        # mask = inputs['attention_mask'].cpu()
        with torch.no_grad():
            outputs = self.drug_model(**inputs)
        
        # 结果转回 CPU 释放显存   
        # drugs = outputs.pooler_output [batch, 768]
        drugs = outputs.last_hidden_state.cpu() # [batch, d_max_tokenLen, 78] [8,222,768]  || [8, 72, 768], [8, 83, 768]
        return drugs
    

    def get_chemberta_max_length(self, all_drug_smiles):
        all_inputs = self.drug_tokenizer(all_drug_smiles, truncation=True)
        # 获取所有编码后的input_ids的长度，取最大值
        max_druglen_all = max([len(x) for x in all_inputs['input_ids']])
        print(f"drug_smiles 的 全局最大 token 长度为: {max_druglen_all}")

        return max_druglen_all



    # CMTarget：提取化合物序列的特征
    # generate drug feature with MorganFingerprint
    def drug_fea_extract(self, drug_sequence):  
        drugs = []

        # 提取1个化合物序列的特征
        if Chem.MolFromSmiles(drug_sequence):
            mol = Chem.MolFromSmiles(drug_sequence)
            radius = 2
            nBits = self.feature_dim
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprint_feature = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0)
        else:
            # print(str(drug))
            # print("Above smile transforms to fingerprint error!!!")
            # print("Please remove!")
            fingerprint_feature = torch.zeros(self.feature_dim, dtype=torch.float32).unsqueeze(0)

        drugs.append(fingerprint_feature)

        
        return drugs
        # drug_feature_lst1 = FeatureExtractor.drug_fea_extract(drug) # fingerprint提取 [1024,];


    

