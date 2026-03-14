import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from embedding.FeatureExtract import *
from torch import nn
from tqdm import tqdm
from pathlib import Path


# --- 重点：自定义 collate_fn 来处理不同长度的 batch ---
def collate_fn(batch):
    proteins, drugs, labels, smiles, sequences = zip(*batch)
    # drugs, proteins, labels = zip(*batch)
    # 将一个 batch 内的序列 pad 到当前 batch 的最大长度
    drugs_pad = pad_sequence(drugs, batch_first=True)
    proteins_pad = pad_sequence(proteins, batch_first=True)
    labels_tensor = torch.stack(labels)
    return drugs_pad, proteins_pad, labels_tensor, smiles, sequences


class DTIDataset(Dataset):
    '''
    输入： 
        df: 包含 (compound, protein, label) 的 pandas DataFrame
    输出：
        dataloader形式 
    '''
    def __init__(self, df):
        # 允许直接接收 DataFrame 对象
        self.data = df.reset_index(drop=True) # 重置索引，防止 slice 后索引不连续导致错误

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 获取数据
        compound = row["compound"]
        protein = row["protein"]
        # 确保 label 是 float32 格式，适合 BCELoss 等损失函数
        label = torch.tensor(row["label"], dtype=torch.float32)

        return compound, protein, label

class FeatureDataset(Dataset):
    def __init__(self, encoder_path, shuffle_csv_path):
        # 加载之前保存的 .pt 特征字典
        self.features = torch.load(encoder_path, map_location="cpu")
        # 加载之前保存的已打乱序列 .csv
        self.df = pd.read_csv(shuffle_csv_path)
        
        # 确保两者的长度一致
        assert len(self.df) == len(self.features["label"]), "特征和序列数量不匹配！"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 获取特征数据
        protein_feat = self.features["protein"][idx]
        drug_feat = self.features["drug"][idx]
        label = self.features["label"][idx]
        
        # 2. 获取原始序列数据 (smiles, sequence)
        smiles = self.df.iloc[idx]['compound']
        sequence = self.df.iloc[idx]['protein']
        
        # 同时返回，保证在一个 Batch 里绝对对齐
        return protein_feat, drug_feat, label, smiles, sequence
    