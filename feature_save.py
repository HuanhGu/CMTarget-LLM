from embedding.dataset_build import *
from embedding.FeatureExtract import FeatureExtractor

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


batch_size = 8
feature_extractor = FeatureExtractor()

def encoder_and_save(df, 
                     encoder_path = "./data/encoder/drugbank_encoder_80pct.pt"):
    d_loader = DataLoader(DTIDataset(df), batch_size=batch_size, shuffle=True)

    all_protein_features = []
    all_drug_features = []
    all_labels = []

    # 使用 tqdm 创建一个进度条实例
    pbar = tqdm(d_loader, desc="Extracting Features", unit="batch")

    with torch.no_grad():
        for compound_batch, protein_batch, label_batch in pbar:        
            p_feats = feature_extractor.pro_fea_extract(protein_batch)
            d_feats = feature_extractor.drug_fea_extract_chemberta(compound_batch)
            # print(p_feats.shape)  #[8, 256, 100]
            # print(d_feats.shape) # [8, 128, 768]
            
            
            # 2. 收集数据 (建议直接转成 CPU tensor 节省内存)
            all_protein_features.append(p_feats.cpu())
            all_drug_features.append(d_feats.cpu())
            all_labels.append(label_batch.cpu())
            
            # 可选：在进度条右侧实时显示当前处理状态
            # pbar.set_postfix({"last_batch_size": len(label_batch)})
            del p_feats, d_feats, compound_batch, protein_batch


    # 4. 循环结束后，统一合并并保存 (只执行一次)
    print("\nPost-processing and saving...")

    all_protein_tensor = torch.cat(all_protein_features, dim=0)
    all_drug_tensor = torch.cat(all_drug_features, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    # labels_tensor = torch.stack(labels)

    torch.save({
        "protein": all_protein_tensor,
        "drug": all_drug_tensor,
        "label": all_labels_tensor
    }, encoder_path)

    print(f"✅ 特征保存完成：{encoder_path} | 总计: {len(all_labels_tensor)} 条数据")

if __name__ == '__main__':
    
    # 1. 加载数据集df
    
    '''
    hit_path="./data/dataset/hit/hit.csv"
    d_df = pd.read_csv(hit_path)
    train_df, test_df = train_test_split(d_df, test_size=0.2, random_state=0, shuffle=True)
    encoder_and_save(train_df, encoder_path = "./data/encoder/hit_encoder_80pct.pt")
    encoder_and_save(test_df, encoder_path = "./data/encoder/hit_encoder_20pct.pt")
    


    drugbank_path="./data/dataset/drugbank/drugbank.csv"
    d_df = pd.read_csv(drugbank_path) 
    train_df, test_df = train_test_split(d_df, test_size=0.2, random_state=0, shuffle=True)
    encoder_and_save(train_df, encoder_path = "./data/encoder/drugbank_encoder_80pct.pt")
    encoder_and_save(test_df, encoder_path = "./data/encoder/drugbank_encoder_20pct.pt")
    '''

    
    dti2_path="./data/dataset/dti2/dti2.csv"
    d_df = pd.read_csv(dti2_path) 
    train_df, test_df = train_test_split(d_df, test_size=0.2, random_state=0, shuffle=True)
    encoder_and_save(train_df, encoder_path = "./data/encoder/dti2_encoder_80pct.pt")
    encoder_and_save(test_df, encoder_path = "./data/encoder/dti2_encoder_20pct.pt")
