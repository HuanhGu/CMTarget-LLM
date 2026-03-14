from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from embedding.FeatureExtract import FeatureExtractor
from embedding.dataset_build import *
from model.CMTargetModel import CMTargetModel
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *
from utils.utils import TrainLogger, PredictLogger, get_data_new_path
from torch.utils.data import TensorDataset, DataLoader


class CMTargetTrainer():
    """
    input:
        dataloader: (compound, protein, label), [3, batch_size, token_num, token_dim]
    
    """
    def __init__(self, configs, source_datapath, model_path):
        self.configs = configs
        self.source_data_path = source_datapath

        self.device = configs['device']
        self.learning_rate = configs['learning_rate']
        self.epochs = configs['epochs']
        self.batch_size = configs['batch_size']

        # self.feature_extractor = feature_extractor
        self.model = self.get_model(model_path)

        
        train_encoder_path = "./data/encoder/drugbank_encoder_80pct.pt"
        test_encoder_path = "./data/encoder/drugbank_encoder_20pct.pt"
        self.train_loader = self.get_dataloader(train_encoder_path)
        self.test_loader = self.get_dataloader(test_encoder_path)

        self.criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_dataloader(self, train_encoder_path):
        checkpoint = torch.load(train_encoder_path)

        # 1. 重新封装成数据集
        dataset = TensorDataset(
            checkpoint["protein"], 
            checkpoint["drug"], 
            checkpoint["label"]
        )

        # 2. 定义新的可遍历对象
        # 这样你可以自由决定加载时的 batch_size，不一定要和保存前一样
        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return val_loader


    '''
    def get_dataloader(self, origin_datapath):
        """划分训练集, 并得到 dataloader [sequence, smiles, label]"""
        df = pd.read_csv(origin_datapath) # [3,3]
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)
        train_seq_loader = DataLoader(DTIDataset(train_df), batch_size=self.batch_size, shuffle=True)
        test_seq_loader = DataLoader(DTIDataset(test_df), batch_size=self.batch_size, shuffle=True)

        return train_seq_loader, test_seq_loader
    '''

    def get_model(self, model_path):
        model = CMTargetModel(self.configs)
        if model_path != '':
            print('Get model from:', model_path)
            model.load_model(model_path)

        return model
    

    def model_train_anepoch(self, model, epoch_id):
        model = model.to(self.device)
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # [smiles, seq, label]
        for protein_batch, compound_batch, label_batch in tqdm(self.train_loader):        

            # 清空梯度
            self.optimizer.zero_grad()

            # 前向传播：三种模态特征对齐融合+MoE编码 in:[3,2,501,100]  [3,2,68,768]
            # outputs是概率值[batch_size,]
            pred_score, contrastive_Loss, load_balancing_loss = model(protein_batch, compound_batch)
            
            # 计算预测损失  [2]  [2,1]
            pred_score = pred_score.cpu()
            pred_loss = self.criterion(pred_score, label_batch)

            # 总损失 = 对比损失 + 负载均衡损失 + 预测损失
            loss = contrastive_Loss + load_balancing_loss + pred_loss

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # 计算准确率
            predicted = (pred_score > 0.5).float()  # 将输出转换为0或1
            correct += (predicted == label_batch).sum().item()
            total += label_batch.size(0)

        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct / total * 100
        print(f"Epoch [{epoch_id+1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%") 
        return avg_loss



    def model_evaluate_anepoch(self, evl_model, epoch_id):
        evl_model = evl_model.to(self.device)
        evl_model.eval()

        targets, predicts = list(), list()
        threshold = 0.5
        with torch.no_grad():
            y_true = []
            y_score = []
            i = 1
            total = len(self.test_loader)
            loop = tqdm(self.test_loader, total=total, smoothing=0, mininterval=1.0)

            for protein_batch, compound_batch, label_batch in loop:
                # 预测结果：三种模态特征对齐融合+MoE编码 in:[3,2,501,100]  [3,2,68,768]
                pred_score, contrastive_Loss, load_balancing_loss = evl_model(protein_batch, compound_batch)              
                pred_score = pred_score.cpu()
                pred = torch.where(pred_score > threshold, torch.tensor(1.0), torch.tensor(0.0))
                
                # 预测list 和  真值list
                targets.extend(label_batch.tolist())
                predicts.extend(pred.tolist())
                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                # 评价指标
                recall, precision, f1, accuracy, auc = calculate_metrics(arr_targets, arr_predicts)
                
                loop.set_description(f'Batch [{i}/{total}]')
                loop.set_postfix(recall=round(recall, 4), precision=round(precision, 4), f1=round(f1, 4),
                                 accuracy=round(accuracy, 4), auc=round(auc, 4))
                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()

        return recall, precision, f1, accuracy, auc, y_true, y_score



    def train(self, output_path):
        print("🚀 start pre-training...")

        # drug_list = self.train_loader.dataset.data['compound'].tolist() + self.test_loader.dataset.data['compound'].tolist()
        # protein_list = self.train_loader.dataset.data['protein'].tolist() + self.test_loader.dataset.data['protein'].tolist()

        logger = TrainLogger(f"Training", self.configs['timestamp'])
        # logger.update_protein_drug(protein_list, drug_list)

        patience = self.configs['patience']
        checkpoint_interval = self.configs['checkpoint_interval']

        max_f1 = 0
        wait = 0  # 用于早停计数器

        for i in range(self.epochs):
            loss = self.model_train_anepoch(self.model, i)
            recall, precision, f1, accuracy, auc, y_true, y_score = self.model_evaluate_anepoch(self.model, i)
            
            logger.write(f"Epoch [{i + 1}/{self.epochs}]: loss = {round(loss, 4)}, recall = {round(recall, 4)}, precision = {round(precision, 4)}, f1 = {round(f1, 4)}, accuracy = {round(accuracy, 4)}, auc = {round(auc, 4)}")
            logger.log_loss(loss)
            logger.log_metrix(recall, precision, f1, accuracy, auc)
            
            if f1 > max_f1:
                logger.update_true_score(y_true, y_score)
                max_f1 = f1
                wait = 0  # 重置等待计数器
                self.model.save_model(output_path)
            else:
                wait += 1
                # print(f"pretrain : No improvement in F1 for {wait} epoch(s).")
            
            if wait >= patience:
                print(f"📊Early stopping triggered. Best F1: {max_f1}")
                break
            
            # checkpoint 保存
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = f"./checkpoints/{self.configs['timestamp']}/pretrain_checkpoint_epoch{i+1}.pt"
                self.model.save_model(checkpoint_path)
                print(f"Checkpoint saved at epoch {i+1} to {checkpoint_path}")


        print(f"\n✅preTraining finished, model has been saved to {output_path}")
        
