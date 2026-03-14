import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader

from embedding.dataset_build import *
from embedding.FeatureExtract import FeatureExtractor

from model.scorer import *
from model.CMTargetModel import *
from trainer.CMTargetTrainer import CMTargetTrainer
from predictor.CMTargetPredictor import CMTargetPredictor
from fineTuner.FineTunner import FineTunner
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

import warnings
warnings.filterwarnings("ignore")


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default = "train", 
                        help="the stage:train, predict")

    parser.add_argument('--checkpoint_interval', type=int, default=8)
    parser.add_argument('-bs', '--batch_size', type = int, default = 16)
    parser.add_argument('-ep', '--epochs', type=int, default = 2)
    parser.add_argument('-lr', '--learning_rate', type=float, default = 1)
    parser.add_argument('--patience', type = int, default=8) 
    parser.add_argument('-scW', '--score_way', type=str, default='MF', 
                        help="choose a scorer, MF,GMF,Cosine ")
    
    parser.add_argument('--source_datapath', type = str, default="./data/dataset/drugbank/drugbank.csv")
    parser.add_argument('--target_datapath', default='./data/dataset/hit/hit.csv')

    
    # parser.add_argument('--timestamp', type=str, default = "001")
    parser.add_argument('-emb', '--embedding_dim', type=int, default=512)
    parser.add_argument('-mod', '--model_name', type=str, default = "CMTarget")
    parser.add_argument('--model_path', type = str, default="")
    parser.add_argument('-pTok', '--protein_encoder_Token_num', type=int, default=416)
    # parser.add_argument('-scD', '--score_emb_dim', type = int, default = 256)


    args = parser.parse_args()

    config = {}
    config['batch_size'] = args.batch_size
    config['checkpoint_interval'] = args.checkpoint_interval
    config['emb'] = args.embedding_dim
    config['epochs'] = args.epochs  
    config['learning_rate'] = args.learning_rate

    config['model'] = args.model_name
    config['model_path'] = args.model_path
    
    config['patience']=args.patience
    config['score_way'] = args.score_way
    # config['score_dim'] = args.score_emb_dim
    config['source_datapath'] = args.source_datapath
    config['target_datapath'] = args.target_datapath
    config['task'] = args.task

    config['timestamp'] = timestamp
    
    return config

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    
    # 2. 获取超参数配置 ./configs/config.json
    configs = prepare()
    
    config_dir = os.path.join('configs', configs['timestamp'])
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    config_path = os.path.join(config_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)
    
    configs['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # 3. 训练模型
    # feature_extractor = FeatureExtractor()
    # 中间暂存模型路径
    os.makedirs(f"checkpoints/{configs['timestamp']}", exist_ok=True)
    pretrain_output_path = f"checkpoints/{configs['timestamp']}/pretrain.pt"
    fintune_output_path = f"checkpoints/{configs['timestamp']}/fineTune.pt"

    if configs['task'] == 'train':
        print(f"⚡train model {configs['model']}: epoch: {configs['epochs']}, batch_size: {configs['batch_size']}, lr: {configs['learning_rate']}")

        trainer = CMTargetTrainer(configs, configs['source_datapath'], configs['model_path'])
        trainer.train(pretrain_output_path)
        
        fineTunner = FineTunner(configs, configs['target_datapath'], configs['model_path'])#model
        fineTunner.fineTune(fintune_output_path)    # 加载pre_train完毕后的model_path, 作为初始值
        

    elif configs['task'] == 'predict':
        if not os.path.exists(configs['model_path']):
            print("please make sure the configs['model_path'] is exist." \
            "If it is none, please execute the training phase")
            sys.exit()
        
        predictor = CMTargetPredictor(configs, configs['model_path'])#model
        predictor.predict()


