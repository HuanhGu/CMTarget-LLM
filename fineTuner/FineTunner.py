from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split

from embedding.FeatureExtract import FeatureExtractor
from embedding.dataset_build import *
from model.CMTargetModel import CMTargetModel
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *
from utils.utils import TrainLogger, PredictLogger, get_data_new_path, MultiTaskLossWrapper

from peft import LoraConfig, get_peft_model
from torch.utils.data import TensorDataset, DataLoader

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
    def __init__(self, configs, target_datapath, model_path):
        self.configs = configs
        self.source_data_path = target_datapath

        self.device = configs['device']
        self.learning_rate = configs['tune_learning_rate']
        self.epochs = configs['epochs_tune']
        self.batch_size = configs['batch_size']

        # self.feature_extractor = feature_extractor
        self.model = self.get_model(model_path)

        train_encoder_path = "./data/encoder/hit_encoder_80pct.pt"
        test_encoder_path = "./data/encoder/hit_encoder_20pct.pt"
        self.train_loader = self.get_dataloader(train_encoder_path)
        self.test_loader = self.get_dataloader(test_encoder_path)

        self.loss_balancer = MultiTaskLossWrapper(task_num=3) # loss均衡器

        self.criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
        self.optimizer = optim.Adam(
            [
                {'params': self.model.parameters()},
                {'params': self.loss_balancer.parameters(), 'lr': 0.01}
            ],
            lr=self.learning_rate
        )


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
    

    def get_model(self, model_path):
        # 1. 初始化原始模型
        model = CMTargetModel(self.configs)
        if model_path != '':
            model.load_model(model_path)

        # 2. 定义 LoRA 配置
        # 以及 Scorer 内部的 pooling 线性层进行微调
        lora_config = LoraConfig(
            r=16,                # 秩大小，可根据显存调整 (8, 16, 32)
            lora_alpha=32,       # 缩放系数，通常为 r 的 2 倍
            target_modules=[
                "tune_linear1",       # 匹配 SelfAttentionPooling 的第一层
                "tune_linear_MF",        # 匹配 MF 打分器的线性层
                "tune_fcn_GMF"            # 匹配 GMF 打分器的线性层
            ],
            lora_dropout=0.1,
            bias="none",
        )


        # 3. 包装模型
        # get_peft_model 会自动冻结所有非 LoRA 参数
        model = get_peft_model(model, lora_config)
        
        '''
        # 4. (可选) 强制确保特定的层完全冻结
        # 1. 首先冻结模型中所有的参数
        for param in model.parameters():
            param.requires_grad = False

        # 2. 专门开启 self.scorer 的参数梯度
        for param in model.scorer.parameters():
            param.requires_grad = True
        '''

        # 查看哪些参数被解冻
        # for name, param in model.named_parameters():
            # print(name, param.requires_grad)

        model.print_trainable_parameters()
        return model


    def model_train_anepoch(self, model, epoch_id):
        model = model.to(self.device)
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for protein_batch, compound_batch, label_batch in tqdm(self.train_loader):        

            # 清空梯度
            self.optimizer.zero_grad()

            # 前向传播：三种模态特征对齐融合+MoE编码 in:[3,2,501,100]  [3,2,68,768]
            # outputs是概率值[batch_size,]
            pred_score, contrastive_Loss, load_balancing_loss = model(protein_batch, compound_batch)
            
            # 计算预测损失  [2]  [2,1]
            # label = label_batch.unsqueeze(1)  # 确保标签是一个列向量
            # label = label.squeeze(1) #label:[2,1] → [2]
            pred_score = pred_score.cpu()
            pred_loss = self.criterion(pred_score, label_batch)

            # 总损失 = 对比损失 + 负载均衡损失 + 预测损失
            # loss = contrastive_Loss + load_balancing_loss + pred_loss
            loss = self.loss_balancer(contrastive_Loss, load_balancing_loss, pred_loss)

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



    def fineTune(self, output_path):
        print("\n🚀 start fine-Tuning...")

        logger = TrainLogger(f"FineTuning", self.configs['timestamp'])
        # logger.update_protein_drug(protein_list, drug_list)

        patience = self.configs['patience']
        checkpoint_interval = self.configs['checkpoint_interval']

        max_f1 = 0
        wait = 0  # 用于早停计数器

        for i in range(self.epochs):
            # print(f"\n the train epoch is : {i} \n")
            loss = self.model_train_anepoch(self.model, i)
            recall, precision, f1, accuracy, auc, y_true, y_score = self.model_evaluate_anepoch(self.model, i)
            
            logger.write(f"Epoch [{i + 1}/{self.epochs}]: loss = {round(loss, 4)}, recall = {round(recall, 4)}, precision = {round(precision, 4)}, f1 = {round(f1, 4)}, accuracy = {round(accuracy, 4)}, auc = {round(auc, 4)}")
            logger.log_loss(loss)
            logger.log_metrix(recall, precision, f1, accuracy, auc)
            
            if f1 > max_f1:
                logger.update_true_score(y_true, y_score)
                wait = 0  # 重置等待计数器
                max_f1 = f1
                self.model.save_model(output_path)
            else:
                wait += 1


            if wait >= patience:
                print(f"Early stopping triggered. Best F1: {max_f1}")
                break
            
            # checkpoint 保存
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = f"./checkpoints/{self.configs['timestamp']}/finTune_checkpoint_epoch{i+1}.pt"
                self.model.save_model(checkpoint_path)
                print(f"Checkpoint saved at epoch {i+1} to {checkpoint_path}")

        print(f"\n✅fine-Tuning finished, model has been saved to {output_path}")