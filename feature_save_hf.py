import h5py
from embedding.dataset_build import *
from embedding.FeatureExtract import FeatureExtractor

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


batch_size = 8
feature_extractor = FeatureExtractor()

"""
提前获取最大token数量, feat_emb时做截断. 
"""
def encoder_and_save(df, encoder_path = "./data/encoder/drugbank_encoder_80pct.h5"):
    # 获取所有化合物序列的最大token数量
    d_max_tokenLen = feature_extractor.get_chemberta_max_length(df['compound'].tolist()) #222
    # p_mean_kmers, p_max_kmers = feature_extractor.get_protein_max_kmers(df['protein'].tolist())
    p_max_tokenLen = feature_extractor.get_probert_max_length(df['protein'].tolist())
    # d_max_tokenLen = 512
    # p_max_kmers = 1024
 
    d_loader = DataLoader(DTIDataset(df), batch_size=batch_size, shuffle=True)
    with h5py.File(encoder_path, "w") as f:
        f.create_dataset("protein", shape=(0, p_max_tokenLen, 1024), maxshape=(None, p_max_tokenLen, 1024), 
                    chunks=(batch_size, p_max_tokenLen, 100), dtype='float32',shuffle=True)
        # f.create_dataset("protein", shape=(0, p_mean_kmers, 100), maxshape=(None, p_mean_kmers, 100), 
                        #  chunks=(batch_size, p_mean_kmers, 100), dtype='float32',shuffle=True)
        f.create_dataset("drug", shape=(0, d_max_tokenLen, 768), maxshape=(None, d_max_tokenLen, 768), 
                         chunks=(batch_size, d_max_tokenLen, 100), dtype='float32',shuffle=True)
        f.create_dataset("label", shape=(0,), maxshape=(None,), chunks=(batch_size * 4,), dtype='int32')

        # 使用 tqdm 包装 d_loader，desc 是进度条前的文字
        pbar = tqdm(enumerate(d_loader), total=len(d_loader), desc="Feature Extracting")

        for batch_idx, (compound_batch, protein_batch, label_batch) in pbar:
            # 提取特征
            # p_feats = feature_extractor.pro_fea_extract(protein_batch, p_mean_kmers).cpu().numpy()
            p_feats = feature_extractor.pro_fea_extract_probert(protein_batch, p_max_tokenLen).cpu().numpy()
            d_feats = feature_extractor.drug_fea_extract_chemberta(compound_batch, d_max_tokenLen).cpu().numpy()
            labels = label_batch.cpu().numpy()
            
            # 追加写入
            for name, data in zip(["protein", "drug", "label"], [p_feats, d_feats, labels]):
                dataset = f[name]
                dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
                dataset[-data.shape[0]:] = data

    print(f"✅ 特征保存完成：{encoder_path} | 总计: {(batch_idx+1)*8} 条数据")


if __name__ == '__main__':

    # dataname = 'drugbank'  #'dti2' 'drugbank' 'hit'
    dataname = 'hit'
    
    # 1. 读取序列数据集, 并划分
    csv_path = Path('data') / 'dataset' / dataname / f'{dataname}.csv'
    d_df = pd.read_csv(csv_path) 
    train_df, test_df = train_test_split(d_df, test_size=0.2, random_state=0, shuffle=True)


    # 2. 获取encoder保存路径
    # 知道dataname, 数据集保存到./data/encoder/ <dataname> /encoder_80.pt
    # 知道dataname, 数据集保存到./data/encoder/ <dataname> /encoder_20.pt
    encoder_dir = Path('data') / 'encoder' / dataname
    encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder_path_80 = encoder_dir / 'encoder_80pct.h5'
    encoder_path_20 = encoder_dir / 'encoder_20pct.h5'
    

    # 3. encoder并保存   
    if not encoder_path_80.exists():
        encoder_and_save(train_df, encoder_path_80)
    if not encoder_path_20.exists():
        encoder_and_save(test_df, encoder_path_20)


"""
把chembert下载到本地
huggingface-cli download --resume-download seyonec/ChemBERTa-zinc-base-v1 --local-dir ./embedding/ChemBERTa
"""