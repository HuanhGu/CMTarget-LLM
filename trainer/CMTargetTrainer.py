""" 只使用bert_tokenier分词表 """
from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import h5py
from torchinfo import summary
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from embedding.FeatureExtract import FeatureExtractor
from embedding.dataset import *
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
    def __init__(self, configs, source_name, model_path):
        self.configs = configs
        self.device = configs['device']
        self.learning_rate = configs['learning_rate_pretrain']
        self.epochs = configs['epochs_train']
        self.batch_size = configs['batch_size']
        self.patience = configs['patience']
        self.checkpoint_interval = configs['checkpoint_interval']
        self.use_selfatt = configs['use_selfatt']
        self.use_moe = configs['use_moe']

        self.model = self.get_model(model_path)
        print(self.model)
        # self.feature_extracror = FeatureExtractor()
        self.train_loader,self.test_loader = self.get_dataloader(source_name) #样本 3599, 29


        print("some settings...")
        self.loss_balancer = MultiTaskLossWrapper(task_num=2).to(self.device) # loss均衡器[只写在这里不可训练，必须加到优化器里]
        # self.criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数  必须用signomid
        self.criterion = nn.BCEWithLogitsLoss()  # 不用sigmoid
        

        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        balancer_params = list(self.loss_balancer.parameters())

        self.optimizer = optim.AdamW([
            {'params': weight_p, 'weight_decay': 1e-4}, 
            {'params': bias_p, 'weight_decay': 0},
            {'params': balancer_params, 'lr': self.learning_rate} # 给 balancer 专门开一组
        ], lr=self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate, max_lr=self.learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=len(self.train_loader))

        
    def get_dataloader(self, dataname = 'hit'):
            """ 得到分词结果 """

            csv_dir = Path('data/dataset')  / dataname 

            train_df = pd.read_csv(csv_dir / 'train.csv')
            train_dataset = DTIDataset(train_df)
            train_size = len(train_dataset)

            test_df = pd.read_csv(csv_dir / 'test.csv')
            test_dataset = DTIDataset(test_df)
            test_size = len(test_dataset)

            train_load = DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=False, num_workers=0)
            test_load = DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=False, num_workers=0)
            
            print(f"总数据数目:{train_size+test_size}, 训练集数目:{train_size}, 测试集数目:{test_size}.")
            
            return train_load, test_load
    
            " 1. 读取序列数据集 "
            "特征提取"
            "数据集划分"
            "得到dataloader"

    def get_model(self, model_path):
        model = CMTargetModel(self.configs)
        if model_path != '':
            print('Get model from:', model_path)
            model.load_model(model_path)
        else:
            #-weights 初始化
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        return model
    
    def get_loss(self, contrastive_Loss, load_balancing_loss, pred_loss):
        "计算损失:  # 总损失 = 对比损失 + 负载均衡损失 + 预测损失"
        if self.use_moe:
            loss_list = torch.stack([load_balancing_loss*0.05, pred_loss])
            loss = self.loss_balancer(loss_list)
        else:
            loss = pred_loss
        return loss

    def model_train_anepoch(self, model, epoch_id):
        model = model.to(self.device)
        model.train()
        
        running_loss = []
        correct = 0
        total = 0
        
        # [smiles, seq, label]
        pbar = tqdm(self.train_loader, desc="🚂 Training", leave=True, ncols=100)
        for compound_batch, protein_batch, label_batch in pbar:        

            # 清空梯度
            self.optimizer.zero_grad()

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
        print(f"Train Epoch [{epoch_id+1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%") 
        return avg_loss, accuracy



    def model_evaluate_anepoch(self, evl_model, epoch_id):
        evl_model = evl_model.to(self.device)
        evl_model.eval()

        threshold = 0.5
        y_true, y_score, running_loss, targets, predicts =[], [], [], [], []
        # 初始化一个包含所有指标名的字典，值为空列表
        metrics = {name: [] for name in ["recall", "precision", "f1", "accuracy", "auc"]}

        with torch.no_grad():
            i = 1
            total = len(self.test_loader)-1  #305
            loop = tqdm(self.test_loader, total=total, desc="Evaluate_An_Epoch",
                        position=0, leave=True,ncols=100,ascii=False)
            #  smoothing=0, mininterval=1.0,

            for compound_batch, protein_batch, label_batch in loop:
                # 预测结果：三种模态特征对齐融合+MoE编码
                logits, contrastive_Loss, load_balancing_loss = evl_model(protein_batch, compound_batch)              
                label_batch = label_batch.to(self.device)
                pred_loss = self.criterion(logits, label_batch)
                # 如果你外接了 nn.BCEWithLogitsLoss（它内部带 Sigmoid），
                # 那么 $\text{Sigmoid}(0.2) \approx 0.55$，$\text{Sigmoid}(0.7) \approx 0.67$。
                
                loss = self.get_loss(contrastive_Loss, load_balancing_loss, pred_loss)
                running_loss.append(loss.item())

                pred_score = torch.sigmoid(logits)
                pred = (pred_score > 0.5).float()  # 将输出转换为0或1
                # 预测list 和  真值list
                targets.extend(label_batch.tolist())
                predicts.extend(pred.tolist())
                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                # 评价指标_这里的roc有问题输入应该是概率
                # recall, precision, f1, accuracy, auc = calculate_metrics(arr_targets, arr_predicts)                
                # loop.set_description(f'Evaluate Batch [{i-1}/{total}]')
                # loop.set_postfix(loss=f"{loss.item():.4f}", f1=round(f1, 4),
                #     recall=round(recall, 4), pre=round(precision, 4), 
                #     acc=round(accuracy, 4), auc=round(auc, 4))
                results = calculate_metrics(arr_targets, arr_predicts)
                metric_names = ["recall", "precision", "f1", "accuracy", "auc"]  
                # 批量存入字典,用于计算后续平均值            
                for name, val in zip(metric_names, results):
                    metrics[name].append(val)
                # 当前值
                current_metrics = dict(zip(metric_names, results)) 
                loop.set_description(f'Evaluate metrics:')
                loop.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    f1=round(current_metrics['f1'], 4),
                    recall=round(current_metrics['recall'], 4), 
                    pre=round(current_metrics['precision'], 4), 
                    acc=round(current_metrics['accuracy'], 4), 
                    auc=round(current_metrics['auc'], 4)
                )
                
                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()
            avg_loss = np.average(running_loss)
            avg_metrics = {name: sum(values)/len(values) for name, values in metrics.items()}

        return avg_metrics['recall'], avg_metrics['precision'], avg_metrics['f1'], avg_metrics['accuracy'], avg_metrics['auc'], \
                y_true, y_score, avg_loss



    def train(self, output_path):
        print("🚀 start pre-training...")
        logger = TrainLogger(f"Training", self.configs['timestamp'])
    
        max_f1 = 0
        wait = 0  # 用于早停计数器

        for i in range(self.epochs):
            # 模型训练与评估
            loss, accuracys = self.model_train_anepoch(self.model, i)
            recall, precision, f1, accuracy, auc, y_true, y_score, test_loss = self.model_evaluate_anepoch(self.model, i)
            
            # 日志
            logger.write(f"Epoch [{i + 1}/{self.epochs}] Train_loss = {round(loss, 4)}, acc = {round(accuracys, 4)}")
            logger.write(f"Epoch [{i + 1}/{self.epochs}] Test_loss = {round(test_loss, 4)}, recall = {round(recall, 4)}, precision = {round(precision, 4)}, f1 = {round(f1, 4)}, accuracy = {round(accuracy, 4)}, auc = {round(auc, 4)}\n")
            logger.log_loss(loss, test_loss)
            logger.log_metrix(recall, precision, f1, accuracy, auc)
            
            # 保存最优模型 & 早停 : f1最大
            if f1 > max_f1:
                logger.update_true_score(y_true, y_score)
                max_f1 = f1
                wait = 0  
                self.model.save_model(output_path)
            else:
                wait += 1
            if wait >= self.patience:
                print(f"📊 Early stopping triggered. Best F1: {max_f1}")
                break
            
            # 每隔一定轮数, 保存 checkpoint
            if (i + 1) % self.checkpoint_interval == 0:
                checkpoint_dir = os.path.join('logs', self.configs['timestamp'], 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"pretrain_checkpoint_epoch{i+1}.pt")
                self.model.save_model(checkpoint_path)
                print(f"Checkpoint saved at epoch {i+1} to {checkpoint_path} 💾")

        print(f"\n✅ preTraining finished, model has been saved to {output_path}")