import h5py
from embedding.dataset_build import *
from embedding.FeatureExtract import FeatureExtractor

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


batch_size = 8
feature_extractor = FeatureExtractor()

def encoder_and_save(df,
                     encoder_path = "./data/encoder/drugbank_encoder_80pct.h5"):

    d_loader = DataLoader(DTIDataset(df), batch_size=batch_size, shuffle=True)

    with h5py.File(encoder_path, "w") as f:
        f.create_dataset("protein", shape=(0, 256, 100), maxshape=(None, 256, 100), chunks=True, dtype='float32')
        f.create_dataset("drug", shape=(0, 128, 768), maxshape=(None, 128, 768), chunks=True, dtype='float32')
        f.create_dataset("label", shape=(0,), maxshape=(None,), chunks=True, dtype='int32')

        # 使用 tqdm 包装 d_loader，desc 是进度条前的文字
        pbar = tqdm(enumerate(d_loader), total=len(d_loader), desc="Feature Extracting")

        total_saved = 0
        for batch_idx, (compound_batch, protein_batch, label_batch) in pbar:
        # for batch_idx, (compound_batch, protein_batch, label_batch) in enumerate(d_loader):
            # 提取特征
            p_feats = feature_extractor.pro_fea_extract(protein_batch).cpu().numpy()
            d_feats = feature_extractor.drug_fea_extract_chemberta(compound_batch).cpu().numpy()
            labels = label_batch.cpu().numpy()

            # 追加写入
            for name, data in zip(["protein", "drug", "label"], [p_feats, d_feats, labels]):
                dataset = f[name]
                dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
                dataset[-data.shape[0]:] = data

    print(f"✅ 特征保存完成：{encoder_path} | 总计: {(batch_idx+1)*8} 条数据")

if __name__ == '__main__':
    
    # 1. 加载数据集df
    dti2_path="./data/dataset/dti2/dti2.csv"
    d_df = pd.read_csv(dti2_path) 
    train_df, test_df = train_test_split(d_df, test_size=0.2, random_state=0, shuffle=True)
    encoder_and_save(train_df, encoder_path = "./data/encoder/dti2_encoder_80pct.h5")
    encoder_and_save(test_df, encoder_path = "./data/encoder/dti2_encoder_20pct.h5")

"""
huggingface-cli download --resume-download seyonec/ChemBERTa-zinc-base-v1 --local-dir ./embedding/ChemBERTa
"""