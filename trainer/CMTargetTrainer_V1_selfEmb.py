from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import h5py
from torchinfo import summary

import pandas as pd
from sklearn.model_selection import train_test_split

from embedding.FeatureExtract import FeatureExtractor
from embedding.dataset_build import *
from model.CMTargetModel import CMTargetModel
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *
from utils.utils import TrainLogger, PredictLogger, MultiTaskLossWrapper
from torch.utils.data import TensorDataset, DataLoader, random_split

class CMTargetTrainer():
    """
    input:
        dataloader: (compound, protein, label), [3, batch_size, token_num, token_dim]
    
    """
    def __init__(self, configs, source_name):
        # self.configs = configs
        self.device = configs['device']
        self.learning_rate = configs['learning_rate_pretrain']
        self.epochs = configs['epochs_train']
        self.batch_size = configs['batch_size']
        self.patience = self.configs['patience']
        self.checkpoint_interval = self.configs['checkpoint_interval']
        
        "数据-从文件加载"
        train_encoder_path = Path('data') / 'encoder' / source_name / 'encoder_80pct.h5'
        test_encoder_path = Path('data') / 'encoder' / source_name / 'encoder_20pct.h5'
        print("📕 get pre-train dataloader.")
        self.train_feat_loader = self.get_dataloader_frompath(train_encoder_path)
        print("📕 get pre-test dataloader.")
        self.test_feat_loader = self.get_dataloader_frompath(test_encoder_path)
        self.val_feat_loader = self.get_dataloader_frompath(test_encoder_path)

        """
        "数据-直接提取"
        feature_extractor = FeatureExtractor()
        self.train_feat_loader, self.test_feat_loader, self.val_feat_loader = self.get_dataloader(feature_extractor, source_name)
        """


        print("some settings...")
        self.loss_balancer = MultiTaskLossWrapper(task_num=3) # loss均衡器
        self.criterion = nn.BCEWithLogitsLoss()  # 不用sigmoid
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate, max_lr=self.learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=max(1, (len(self.train_feat_loader) // self.batch_size)))


    def get_model(self, model_path=''):
        model = CMTargetModel(self.configs)
        if model_path != '':
            print('Get model from:', model_path)
            model.load_model(model_path)
        return model
    
    
    def get_dataloader_frompath(self, train_encoder_path):
        # 判断文件类型
        file_ext = os.path.splitext(train_encoder_path)[-1].lower()

        if file_ext in ['.h5', '.hdf5']:
            with h5py.File(train_encoder_path, "r") as f:
                protein = torch.tensor(f["protein"][:], dtype=torch.float32)
                drug = torch.tensor(f["drug"][:], dtype=torch.float32)
                label = torch.tensor(f["label"][:], dtype=torch.float32)
            dataset = TensorDataset(protein, drug, label)
        elif file_ext in ['.pt', '.pth']:
            checkpoint = torch.load(train_encoder_path)
            dataset = TensorDataset(checkpoint["protein"], checkpoint["drug"], checkpoint["label"])
        else:
            print("there are no encoder files, please execute feature_save.py")

        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return val_loader
    
    def get_dataloader(self, feature_extractor, dataname = 'hit'):
        """ pre-extract :  get feature using chemberta and word2vec. """
        " 1. 读取序列数据集 "
        csv_path = Path('data') / 'dataset' / dataname / f'{dataname}.csv'
        d_df = pd.read_csv(csv_path) 

        "提取特征"
        # 获取所有化合物序列的最大token数量
        d_max_tokenLen = feature_extractor.get_chemberta_max_length(d_df['compound'].tolist()) #222
        p_mean_kmers, p_max_kmers = feature_extractor.get_protein_max_kmers(d_df['protein'].tolist())
        
        p_feats = feature_extractor.pro_fea_extract(d_df['protein'].tolist(), p_mean_kmers).cpu()
        d_feats = feature_extractor.drug_fea_extract_chemberta(d_df['protein'].tolist(), d_max_tokenLen).cpu()
        label_tensor = torch.tensor(d_df['label'].values).cpu()
        feature_dataset = TensorDataset(p_feats, d_feats, label_tensor)

        "数据集划分,得到 feature_dataset, feature_dataloader"
        total_sz = len(d_df)
        train_sz, val_sz, test_sz = int(0.7*total_sz), int(0.1*total_sz), total_sz - int(0.7*total_sz) - int(0.2*total_sz)
        train_dataset, val_dataset, test_dataset = random_split(feature_dataset, [train_sz, val_sz, test_sz])
        train_feat_load = DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True)
        test_feat_load = DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=True)
        val_feat_load = DataLoader(dataset=val_dataset,batch_size=self.batch_size,shuffle=True)

        return train_feat_load, test_feat_load, val_feat_load

    
    def get_loss(self, contrastive_Loss, load_balancing_loss, pred_loss):
        "计算损失:  # 总损失 = 对比损失 + 负载均衡损失 + 预测损失"
        "量级 : [27+2+0.68]"
        # 19 + 2 + 0.6930[27+2+0.68]
        # loss = self.loss_balancer(contrastive_Loss, load_balancing_loss, pred_loss)
        loss = pred_loss  # 量级：0~10s
        return loss
    

    def train_anepoch(self, model, epoch_id):
        model = model.to(self.device)
        model.train()
        
        running_loss = []
        correct = 0
        total = 0

        # [smiles, seq, label]
        pbar = tqdm(self.train_feat_loader, desc="Training", position=0, leave=True, ncols=100)
        for protein_batch, compound_batch, label_batch in pbar:        
            
            self.optimizer.zero_grad() # 清空梯度

            # 前向传播：三种模态特征对齐融合+MoE编码 in:[3,2,501,100]  [3,2,68,768]
            # 1打分, 2损失
            logits, contrastive_Loss, load_balancing_loss = model(protein_batch, compound_batch)
            
            # 计算预测损失  [2]  [2,1]
            label_batch = label_batch.to(self.device)
            pred_loss = self.criterion(logits, label_batch)

            # 总损失 = 对比损失 + 负载均衡损失 + 预测损失 19 + 2 + 0.6930[27+2+0.68]
            loss = self.get_loss(contrastive_Loss, load_balancing_loss, pred_loss)

            # 反向传播和优化
            loss.backward() #计算梯度（找准方向）
            self.optimizer.step() #更新参数
            self.scheduler.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            running_loss.append(loss.item())

            # 计算准确率
            pred_score = torch.sigmoid(logits)
            predicted = (pred_score > 0.5).float()  # 将输出转换为0或1
            correct += (predicted == label_batch).sum().item()
            total += label_batch.size(0)

        avg_loss = np.average(running_loss)
        accuracy = correct / total * 100
        print(f"🚂 Train Epoch [{epoch_id+1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%") 
        return avg_loss



    def val_anepoch(self, val_model, epoch_id):
        val_model = val_model.to(self.device)
        val_model.eval()

        targets, predicts = list(), list()
        threshold = 0.5
        running_loss = []

        with torch.no_grad():
            y_true = []
            y_score = []
            i = 1
            total = len(self.val_feat_loader)-1  #305
            loop = tqdm(self.val_feat_loader, total=total, desc="val_an_epoch",
                        position=0, leave=True,ncols=100,ascii=False)

            for protein_batch, compound_batch, label_batch in loop:
                logits, contrastive_Loss, load_balancing_loss = val_model(protein_batch, compound_batch)              
                pred_score = torch.sigmoid(logits)
                pred_score = pred_score.cpu()
                # predicted = (pred_score > 0.5).float()  # 将输出转换为0或1

                pred_loss = self.criterion(pred_score, label_batch)
                
                loss = self.get_loss(contrastive_Loss, load_balancing_loss, pred_loss)
                running_loss.append(loss.item())

                # pred = torch.where(pred_score > threshold, torch.tensor(1.0), torch.tensor(0.0))
                pred = (pred_score > 0.5).float()  # 将输出转换为0或1
                # 预测list 和  真值list
                targets.extend(label_batch.tolist())
                predicts.extend(pred.tolist())
                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                # 评价指标_这里的roc有问题输入应该是概率
                recall, precision, f1, accuracy, auc = calculate_metrics(arr_targets, arr_predicts)
                
                loop.set_description(f'Evaluate Batch [{i-1}/{total}]')
                loop.set_postfix(loss=f"{loss.item():.4f}", f1=round(f1, 4),
                    recall=round(recall, 4), pre=round(precision, 4), 
                    acc=round(accuracy, 4), auc=round(auc, 4))
                
                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()

            avg_loss = np.average(running_loss)

        return recall, precision, f1, accuracy, auc, y_true, y_score, avg_loss



    def forward(self, best_model_path):
        print("🚀 start pre-training...")

        train_val_model = self.get_model()
        logger = TrainLogger(f"PreTraining", self.configs['timestamp'])
        max_f1 = 0
        wait = 0  # 用于早停计数器
        
        for i in range(self.epochs):
            # 模型 train 与 val
            train_loss = self.train_anepoch(train_val_model, i)
            recall, precision, f1, accuracy, auc, y_true, y_score, val_loss = self.val_anepoch(train_val_model, i)
            
            # 日志
            logger.write(f"Epoch [{i + 1}/{self.epochs}]: trainloss = {round(train_loss, 4)}, valloss={round(val_loss, 4)}, \
                         recall = {round(recall, 4)}, precision = {round(precision, 4)}, f1 = {round(f1, 4)}, accuracy = {round(accuracy, 4)}, auc = {round(auc, 4)}")
            logger.log_loss(train_loss, val_loss)
            logger.log_metrix(recall, precision, f1, accuracy, auc)
            
            # 保存最优模型 : f1最大
            if f1 > max_f1:
                logger.update_true_score(y_true, y_score)
                max_f1 = f1
                wait = 0  
                train_val_model.save_model(best_model_path)
            else:
                wait += 1
                # print(f"pretrain : No improvement in F1 for {wait} epoch(s).")
            
            # 早停
            if wait >= self.patience:
                print(f"📊 Early stopping triggered. Best F1: {max_f1}")
                break
        
        """test获取指标结果"""    
        best_model = self.get_model(best_model_path)
        recall, precision, f1, accuracy, auc, y_true, y_score, test_loss = self.val_anepoch(best_model, i)
        print(f"\n✅ preTraining finished,best model has been saved to {best_model_path}")
        




"""
# 每隔一定轮数, 保存 checkpoint
if (i + 1) % self.checkpoint_interval == 0:
    checkpoint_dir = os.path.join('logs', self.configs['timestamp'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"pretrain_checkpoint_epoch{i+1}.pt")
    train_val_model.save_model(checkpoint_path)
    print(f"Checkpoint saved at epoch {i+1} to {checkpoint_path} 💾")
"""