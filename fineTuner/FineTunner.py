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
        self.patience = configs['patience']
        self.checkpoint_interval = configs['checkpoint_interval']

        self.model = self.get_model(model_path)
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
            {'params': balancer_params, 'lr': self.learning_rate} # 给 balancer 专门开一组
        ], lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate, max_lr=self.learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=len(self.train_loader))



    def get_dataloader(self, dataname="hit"):
            """ 得到分词结果 """

            " 1. 读取序列数据集 "
            csv_path = Path('data') / 'dataset' / dataname / f'{dataname}.csv'
            d_df = pd.read_csv(csv_path) 
            "特征提取"
            full_dataset = DTIDataset(d_df)       # drug,pro,label

            "数据集划分"
            total_size = len(full_dataset)
            train_size = int(0.8 * total_size)
            test_size = total_size - train_size
            train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

            "得到dataloader"
            train_load = DataLoader(dataset=train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=0)
            test_load = DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=True, num_workers=0)
            print(f"总数据数目:{total_size}, 训练集数目:{train_size}, 测试集数目:{test_size}.")

            return train_load, test_load


    def get_model(self, model_path):
        # 1. 初始化原始模型
        model = CMTargetModel(self.configs)
        if model_path != '':
            model.load_model(model_path)

        target_modules = [
            # "W_Q", "W_K", "W_V",
            "gate_proj",'shared_expert_gate','gate.0',
            'd_a','p_a', "tune_linear1"
        ]
        
        # 2. 定义 LoRA 配置
        lora_config = LoraConfig(
            r=16,                # 秩大小，可根据显存调整 (8, 16, 32)
            lora_alpha=32,       # 缩放系数，通常为 r 的 2 倍
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=None  # 你在预测蛋白-药物评分
        )

        # 3. 包装模型
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  
        #trainable params: 49,152 || all params: 28,791,298 || trainable%: 0.1707

        return model


    def get_loss(self, contrastive_Loss, load_balancing_loss, pred_loss):
        "计算损失:  # 总损失 = 对比损失 + 负载均衡损失 + 预测损失"
        loss_list = torch.stack([load_balancing_loss*0.05, pred_loss])
        # print(f"load_balancing_loss:{load_balancing_loss}, pred_loss:{pred_loss}")
        loss = self.loss_balancer(loss_list)
        return loss

    def model_train_anepoch(self, model, epoch_id):
        model = model.to(self.device)
        model.train()
        
        running_loss = []
        correct = 0
        total = 0

        print(f"****** Epoch [{epoch_id+1}/{self.epochs}] *****")
        pbar = tqdm(self.train_loader, desc=f"Train Epoch", position=0, leave=True, ncols=100)
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
                             mloss=f"{load_balancing_loss.item():.4f}",
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
        # 初始化一个包含所有指标名的字典，值为空列表
        metrics = {name: [] for name in ["recall", "precision", "f1", "accuracy", "auc"]}

        with torch.no_grad():
            i = 1
            total = len(self.test_loader)
            loop = tqdm(self.test_loader, total=total, smoothing=0, mininterval=1.0,
                        position=0, leave=True,dynamic_ncols=True,ascii=False)

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
                loop.set_description(f'Evaluate metrics:')
                loop.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    f1=round(current_metrics['f1'], 4),
                    recall=round(current_metrics['recall'], 4), 
                    pre=round(current_metrics['precision'], 4), 
                    acc=round(current_metrics['accuracy'], 4), 
                    auc=round(current_metrics['auc'], 4)
                )
                # bls=f"{load_balancing_loss.item():.4f}",
                # pls= f"{pred_loss.item():.4f}"

                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()

            avg_loss = np.average(running_loss)
            avg_metrics = {name: sum(values)/len(values) for name, values in metrics.items()}

        return avg_metrics['recall'], avg_metrics['precision'], avg_metrics['f1'], avg_metrics['accuracy'], avg_metrics['auc'], \
                y_true, y_score, avg_loss


    
    def fineTune(self, output_path):
        print("\n🚀 start fine-Tuning...")

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


'''
必须加 LoRA：MoE、SelfAttention。
可选加 LoRA：scorer 的 attention 和 pooling 层（有助于微调评分）。
不加 LoRA：embedding 层、输出线性层 linear2、Cosine。


# 重点：target_modules 需要包含模型中所有的线性层关键字
    target_modules_gemini=[
        # 1. 蛋白质与药物的 Attention 部分
        "W_Q", "W_K", "W_V", 
        # 2. MoE 专家系统内的投影层 (Qwen2MoeMLP)
        "gate_proj", "up_proj", "down_proj",
        # 3. MoE 的门控机制
        "gate", "shared_expert_gate",
        # 4. Scorer 评分层中的线性层
        "d_a", "p_a", "tune_linear1"
    ]


from peft import LoraConfig

config = LoraConfig(
    r=16, # LoRA 的秩，可根据显存调整（常用 8, 16, 32）
    lora_alpha=32, # 缩放系数，通常设为 r 的 2 倍
    # 重点：target_modules 需要包含模型中所有的线性层关键字
    target_modules=[
        # 1. 蛋白质与药物的 Attention 部分
        "W_Q", "W_K", "W_V", 
        # 2. MoE 专家系统内的投影层 (Qwen2MoeMLP)
        "gate_proj", "up_proj", "down_proj",
        # 3. MoE 的门控机制
        "gate", "shared_expert_gate",
        # 4. Scorer 评分层中的线性层
        "d_a", "p_a", "tune_linear1"
    ],
    lora_dropout=0.1,
    bias="none", # 默认不训练 bias
    modules_to_save=["scorer.score"], # 如果 Cosine 是可学习的，或者想全参数微调输出层
    task_type=None # 因为是自定义模型，设为 None 或自定义字符串
)
'''