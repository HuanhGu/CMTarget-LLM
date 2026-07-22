from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import h5py
from torchinfo import summary
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from embedding.FeatureExtract import FeatureExtractor
from embedding.dataset import *
from model.CMTargetModel import CMTargetModel
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *
from utils.utils import TrainLogger, PredictLogger, MultiTaskLossWrapper

from peft import LoraConfig, get_peft_model
from torch.utils.data import TensorDataset, DataLoader, random_split

'''
1. ⚠️ 注意：如果 linear2 是输出层（例如 [hidden → 1]），低秩矩阵的作用可能有限，因为矩阵很小。
这种方式是不使用 LoRA，直接微调原始权重
2. get_peft_model 会自动冻结所有非 LoRA 参数
'''



class FineTunner():
    """
    input:
        dataloader: (compound, protein, label), [3, batch_size, token_num, token_dim]
    
    """
    def __init__(self, configs, target_name, model_path):
        self.configs = configs
        self.target_name = target_name
        self.device = configs['device']
        self.learning_rate = configs['learning_rate_tune']
        self.epochs = configs['epochs_tune']
        self.batch_size = configs['batch_size']
        self.patience = configs['patience_tune']
        self.checkpoint_interval = configs['checkpoint_interval']
        
        self.use_selfatt = configs['use_selfatt']
        self.use_moe = configs['use_moe']

        self.model = self.get_model(model_path)
        # print(f"fine-tune model{self.model}")
        self.train_loader,self.test_loader = self.get_dataloader(target_name) #样本 3599, 29

        self.loss_balancer = MultiTaskLossWrapper(task_num=2).to(self.device) # loss均衡器
        self.criterion = nn.BCEWithLogitsLoss()  # 使用二分类交叉熵损失函数
        
        # 优化器
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
            {'params': balancer_params, 'lr': self.learning_rate} # 给 balancer 专门开一组  #*1
        ], lr=self.learning_rate)
        # 配合平滑的学习率策略
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate, max_lr=self.learning_rate * 10,#1
                                                cycle_momentum=False,
                                                step_size_up=2 * len(self.train_loader))  # 通常与sgd配合使用*5



    def get_dataloader(self, dataname="hit"):
            """ 得到分词结果 """

            csv_dir = Path('data/dataset')  / dataname 

            train_df = pd.read_csv(csv_dir / 'train.csv')
            train_df=train_df[:50]
            train_dataset = DTIDataset(train_df)
            train_size = len(train_dataset)

            test_df = pd.read_csv(csv_dir / 'test.csv')
            test_df = test_df[:50]
            test_dataset = DTIDataset(test_df)
            test_size = len(test_dataset)

            train_load = DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=False, num_workers=0)
            test_load = DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=False, num_workers=0)
            
            print(f"总数据数目:{train_size+test_size}, 训练集数目:{train_size}, 测试集数目:{test_size}.")
            
            return train_load, test_load


    def get_model(self, model_path):
        # 1. 初始化原始模型
        model = CMTargetModel(self.configs)
        if model_path != '':
            # 加载预训练模型
            print('Get model from:', model_path)
            model.load_model(model_path)


        # 2. 定义 LoRA 配置
        # 微调层
        target_modules=[
        "W_Q", "W_K", "W_V",    # 1. 蛋白质与药物的 Attention 部分
        "gate_proj","up_proj",  # "down_proj",   # 2. MoE 专家系统内的投影层 (Qwen2MoeMLP)
        # "gate.0", "shared_expert_gate",       # 3. MoE 的门控机制
        # "d_a", "p_a", "tune_linear1", "linear2"       # 4. Scorer 评分层中的线性层
        ]
        # 全训练层
        modules_to_save=[
            # "W_Q", "W_K", "W_V", 
            "down_proj", # "gate_proj", "up_proj",
            "gate.0", "shared_expert_gate","shared_expert.gate_proj","shared_expert.up_proj","shared_expert.down_proj"
            "d_a", "p_a", "tune_linear1", "linear2"
        ]
        lora_config = LoraConfig(
            r=32,                # 秩大小，可根据显存调整 (8, 16, 32)
            lora_alpha=64,       # 缩放系数，通常为 r 的 2 倍
            target_modules=target_modules,
            modules_to_save = modules_to_save,
            lora_dropout=0.1,
            bias="none",
            task_type=None  # 你在预测蛋白-药物评分
        )
        

        # 3. 检查模型中存在的 target_modules
        existing_modules = []
        for name, _ in model.named_modules():
            for tm in target_modules:
                if tm in name:
                    existing_modules.append(tm)

        if not existing_modules:
            print("⚠️ Warning: No target modules found in the model. Skipping LoRA injection.")
            return model  # 直接返回原模型
        else:
            print(f"✅ Applying LoRA to modules: {existing_modules}")
            # 更新配置，只保留实际存在的模块
            lora_config.target_modules = existing_modules
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()      #trainable params: 49,152 || all params: 28,791,298 || trainable%: 0.1707
            
            # print(f"lora_config:{lora_config}")
            return model


    def get_loss(self, contrastive_Loss, load_balancing_loss, pred_loss):
        "计算损失:  # 总损失 = 对比损失 + 负载均衡损失 + 预测损失"
        if self.use_moe:
            # print(f"load_balancing_loss:{load_balancing_loss}, pred_loss:{pred_loss}")
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
        is_atty = sys.stdout.isatty()

        print(f"—————————————————— epoch [{epoch_id+1}/{self.epochs}] ——————————————————")
        pbar = tqdm(self.train_loader, desc=f"Train Epoch", position=0, leave=True, ncols=100,
                    mininterval=1,  disable=not is_atty) 
        for compound_batch, protein_batch, label_batch in pbar:
            self.optimizer.zero_grad()

            # 前向传播：特征对齐+MoE编码 , outputs概率
            logits, contrastive_Loss, load_balancing_loss = model(protein_batch, compound_batch)
            # 损失loss
            label_batch = label_batch.to(self.device)
            pred_loss = self.criterion(logits, label_batch)
            loss = self.get_loss(contrastive_Loss, load_balancing_loss, pred_loss)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            pbar.set_postfix(loss= f"{loss.item():.4f}", 
                            #  mloss=f"{load_balancing_loss.item():.4f}",
                             ploss= f"{pred_loss.item():.4f}")
            running_loss.append(loss.item())

            # 计算准确率
            pred_score = torch.sigmoid(logits)
            predicted = (pred_score > 0.5).float()  # 将输出转换为0或1
            correct += (predicted == label_batch).sum().item()
            total += label_batch.size(0)

        avg_loss = np.average(running_loss)
        accuracy = correct / total * 100
        print(f"Train metric: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%") 
        return avg_loss, accuracy



    def model_evaluate_anepoch(self, evl_model, epoch_id):
        evl_model = evl_model.to(self.device)
        evl_model.eval()

        y_true, y_score, running_loss, targets, predicts =[], [], [], [], []
        metrics = {name: [] for name in ["recall", "precision", "f1", "accuracy", "auc"]}

        is_atty = sys.stdout.isatty()
        with torch.no_grad():
            i = 1
            loop = tqdm(self.test_loader, smoothing=0, position=0, leave=True,
                        mininterval=1, disable=not is_atty)
            
            for compound_batch, protein_batch, label_batch in loop:
                # 预测结果：三种模态特征对齐融合+MoE编码 in:[3,2,501,100]  [3,2,68,768]
                logits, contrastive_Loss, load_balancing_loss = evl_model(protein_batch, compound_batch)              
                
                # loss
                label_batch = label_batch.to(self.device)
                pred_loss = self.criterion(logits, label_batch)
                loss = self.get_loss(contrastive_Loss, load_balancing_loss, pred_loss)
                running_loss.append(loss.item())

                pred_score = torch.sigmoid(logits)
                pred = (pred_score > 0.5).float()
                
                # 预测list 和  真值list
                targets.extend(label_batch.tolist())
                predicts.extend(pred.tolist())
                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                "评价指标"
                results = calculate_metrics(arr_targets, arr_predicts)
                metric_names = ["recall", "precision", "f1", "accuracy", "auc"]  
                # 批量存入字典,用于计算后续平均值            
                for name, val in zip(metric_names, results):
                    metrics[name].append(val)
                # 当前值
                current_metrics = dict(zip(metric_names, results)) 
                loop.set_description(f'🚂Evaluating')
                loop.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    # f1=round(current_metrics['f1'], 4),
                    # recall=round(current_metrics['recall'], 4), 
                    # pre=round(current_metrics['precision'], 4), 
                    # acc=round(current_metrics['accuracy'], 4), 
                    # auc=round(current_metrics['auc'], 4)
                )
                # bls=f"{load_balancing_loss.item():.4f}",
                # pls= f"{pred_loss.item():.4f}"

                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()

            avg_loss = np.average(running_loss)
            avg_metrics = {name: sum(values)/len(values) for name, values in metrics.items()}
            out_str = ", ".join([f"{name}: {avg_metrics[name]:.4f}" for name in metric_names])
            print(f"Evaluate Epoch [{epoch_id+1}/{self.epochs}], avg_loss= {avg_loss:.4f}, {out_str}") 

        return avg_metrics['recall'], avg_metrics['precision'], avg_metrics['f1'], avg_metrics['accuracy'], avg_metrics['auc'], \
                y_true, y_score, avg_loss


    
    def fineTune(self, output_path):
        print("🚀 start fine-Tuning...")

        logger = TrainLogger(f"FineTuning", self.configs['timestamp'])
        max_f1 = 0
        wait = 0  # 用于早停计数器

        for i in range(self.epochs):
            # print(f"\n the train epoch is : {i} \n")
            loss, accuracys = self.model_train_anepoch(self.model, i)
            recall, precision, f1, accuracy, auc, y_true, y_score, test_loss = self.model_evaluate_anepoch(self.model, i)


            logger.write(f"Epoch [{i + 1}/{self.epochs}] Train_loss = {round(loss, 4)}, acc = {round(accuracys, 4)}")
            logger.write(f"Epoch [{i + 1}/{self.epochs}] Test_loss = {round(test_loss, 4)}, recall = {round(recall, 4)}, precision = {round(precision, 4)}, f1 = {round(f1, 4)}, accuracy = {round(accuracy, 4)}, auc = {round(auc, 4)}\n")
            logger.log_loss(loss, test_loss)
            logger.log_metrix(recall, precision, f1, accuracy, auc)
            
            if f1 > max_f1:
                print(f"f1 was increased from {max_f1} to {f1} in epoch{i}.")
                logger.update_true_score(y_true, y_score)
                wait = 0  # 重置等待计数器
                max_f1 = f1
                self.model.save_model(output_path)
            else:
                wait += 1

            if wait >= self.patience:
                print(f"📊 Early stopping triggered. Best F1: {max_f1}")
                break
            
            # checkpoint 保存
            if (i + 1) % self.checkpoint_interval == 0:
                checkpoint_dir = os.path.join('logs', self.configs['timestamp'], 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"fintune_checkpoint_epoch{i+1}.pt")

                self.model.save_model(checkpoint_path)
                print(f"Checkpoint saved at epoch {i+1} to {checkpoint_path}💾")

        print(f"\n✅ fine-Tuning finished, model has been saved to {output_path}")
