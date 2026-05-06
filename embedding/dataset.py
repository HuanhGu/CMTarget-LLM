import torch
from torch.utils.data import Dataset
import numpy as np
from embedding.FeatureExtract import FeatureExtractor

class DTIDataset(Dataset):
    '''
    输入[df] : 包含 (compound, protein, label) 的 pandas DataFrame
    '''
    def __init__(self, d_df):
        self.fe = FeatureExtractor()
        # 获取所有化合物序列的最大token数量
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
    