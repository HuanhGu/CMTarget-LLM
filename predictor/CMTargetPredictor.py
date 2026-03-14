import ast
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from tqdm import tqdm
import pandas as pd
from utils.metrix import calculate_metrics
from utils.utils import PredictLogger, get_data_new_path

from embedding.dataset_build import FeatureExtractor, collate_fn, DTIDataset
from embedding.FeatureExtract import *
from model.CMTargetModel import *
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *


class CMTargetPredictor():
    def __init__(self, configs, pred_datapath, model_path):
        self.configs = configs
        self.datapath = pred_datapath
        self.batch_size = configs['batch_size']

        self.device = configs['device']
        # self.feature_extractor = feature_extractor

        
        pred_datapath = "./data/encoder/hit_encoder_20pct.pt"

        self.pred_dataloader = self.get_dataloader(pred_datapath)
        self.model = self.get_model(model_path)

        
    
    def get_model(self, model_path):
        model = CMTargetModel(self.configs)
        if model_path != '':
            print('Get model from:', model_path)
            model.load_model(model_path)

        return model

    
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

    
    # recall, precision, f1, accuracy, auc, y_true, y_score = self.pred_anepoch(self.model, self.pred_dataloader)
    def pred_anepoch(self, pred_model, pred_dataloader):
        pred_model = pred_model.to(self.device)
        pred_model.eval()
        targets, predicts = list(), list()
        threshold = 0.5
        with torch.no_grad():
            y_true = []
            y_score = []
            i = 1
            total = len(pred_dataloader)
            loop = tqdm(pred_dataloader, total = total, smoothing=0, mininterval=1.0)

            for protein_batch, compound_batch, label_batch in loop:
                
                #预测结果
                pred_score, _, _ = pred_model(protein_batch, compound_batch)
                pred_score = pred_score.cpu()
                pred = torch.where(pred_score > threshold, torch.tensor(1.0), torch.tensor(0.0))

                # 预测list 和 真值 list
                targets.extend(label_batch.tolist())
                predicts.extend(pred.tolist())
                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                # 评价指标
                recall, precision, f1, accuracy, auc = calculate_metrics(arr_targets, arr_predicts)

                loop.set_description(f'Batch [{i}/{total}]')
                # loop.set_postfix(recall=round(recall, 4), precision=round(precision, 4), f1=round(f1, 4),
                                #  accuracy=round(accuracy, 4), auc=round(auc, 4))
                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()

        return recall, precision, f1, accuracy, auc, y_true, y_score



    def predict(self):
        print("start Predicting")
        # self.pred_dataloader, self.pred_ds_smiles = self.get_dataloader()
        # self.model = self.get_model()
        
        # protein_list = self.pred_dataloader.dataset.data['compound'].tolist() 
        # drug_list = self.pred_dataloader.dataset.data['protein'].tolist() 

        logger = PredictLogger(f"Predicting", self.configs['timestamp'])
        # logger.update_protein_drug(protein_list, drug_list)
        
        # 评估结果
        recall, precision, f1, accuracy, auc, y_true, y_score = self.pred_anepoch(self.model, self.pred_dataloader)
        
        # logger.update_protein_drug(protein_list, drug_list)
        logger.update_true_score(y_true, y_score)
        logger.log_metrix(recall, precision, f1, accuracy, auc)

        print("\n✅Predicting finished. Please see logs.")
        