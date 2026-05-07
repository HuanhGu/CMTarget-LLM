import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime

# from embedding.dataset_build import *
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

    parser.add_argument('-bs', '--batch_size', type = int, default = 64)#128

    parser.add_argument('--checkpoint_interval', type=int, default=30)
    parser.add_argument('-eptr', '--epochs_train', type=int, default = 300)#
    parser.add_argument('-eptu', '--epochs_tune', type=int, default = 200)#
    parser.add_argument('-lrp', '--learning_rate_pretrain', type=float, default = 2e-5)
    parser.add_argument('-lrt', '--learning_rate_tune', type=float, default = 2e-6) # 2e-5 微调学习率应该更大还是更小？更小，约1/10 or 1/100
    parser.add_argument('-mod', '--model_name', type=str, default = "CMTarget")
    parser.add_argument('--model_path', type = str, default="./data/models/pretrain.pt")
    
    parser.add_argument('--patience', type = int, default=30) 
    parser.add_argument('-score', '--score_way', type=str, default='Cosine', 
                        help="choose a scorer, MF,GMF,Cosine ")
    
    parser.add_argument('-s', '--source_name', type = str, default="drugbank")#drugbank
    parser.add_argument('-t', '--target_name', default='hit')#hit
    
    parser.add_argument('--token_dim_pro', type = int, default='512')#probert=1024, w2c=100
    parser.add_argument('--token_dim_drug', type = int, default='512')#chemberta=768
    parser.add_argument('--task', type=str, default = "finetune", 
                        help="choose the stage : train, finetune, predict")

    parser.add_argument('-m', '--remark', type=str, default = "lra_r = 16, bz=64. tune:from down_proj to end.")
 
    args = parser.parse_args()

    config = {}
    config['batch_size'] = args.batch_size
    config['checkpoint_interval'] = args.checkpoint_interval

    config['epochs_train'] = args.epochs_train  
    config['epochs_tune'] = args.epochs_tune 
    config['learning_rate_pretrain'] = args.learning_rate_pretrain
    config['learning_rate_tune'] = args.learning_rate_tune

    config['model'] = args.model_name
    config['model_path'] = args.model_path
    config['patience']=args.patience

    config['score_way'] = args.score_way
    # config['score_dim'] = args.score_emb_dim
    config['source_name'] = args.source_name
    config['target_name'] = args.target_name
    config['task'] = args.task
    config['timestamp'] = timestamp
    config['token_dim_pro'] = args.token_dim_pro
    config['token_dim_drug'] = args.token_dim_drug

    config['remark'] = args.remark
    
    return config

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    
    # 2. 获取超参数配置 ./configs/config.json
    configs = prepare()

    config_dir = Path('logs') / configs['timestamp'] / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)
    configs['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # 3. 训练模型
    # 中间暂存模型路径
    model_output_dir = Path("logs") / configs['timestamp'] /"checkpoints" 
    os.makedirs(model_output_dir, exist_ok=True)
    pretrain_output_path = model_output_dir / "pretrain.pt"
    fintune_output_path = model_output_dir / "fineTune.pt"
    
    start = datetime.now()

    if configs['task'] == 'train':
        print(
        f"⚡[train model {configs['model']}]\n"
        f"  batch_size: {configs['batch_size']}\n"
        f"  epochs: {configs['epochs_train']} (pre) / {configs['epochs_tune']} (tune)\n"
        f"  lr: {configs['learning_rate_pretrain']} (pre) / {configs['learning_rate_tune']} (tune)"
        )
        # 源域训练
        trainer = CMTargetTrainer(configs, configs['source_name'], configs['model_path'])
        trainer.train(pretrain_output_path)
    
    elif configs['task'] == 'finetune':
        # 目标域微调
        fineTunner = FineTunner(configs, configs['target_name'], configs['model_path'])#model
        fineTunner.fineTune(fintune_output_path)    # 加载pre_train完毕后的model_path, 作为初始值
        

    elif configs['task'] == 'predict':
        if not os.path.exists(configs['model_path']):
            print("please make sure the configs['model_path'] is exist." \
            "If it is none, please execute the training phase")
            sys.exit()
        
        predictor = CMTargetPredictor(configs, configs['model_path'])#model
        predictor.predict()

    end = datetime.now()
    print("-" * 30)
    print(f"Pre-Train Start Time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pre-Train End Time:   {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration:       {end-start}")
    print("-" * 30)



"""
nohup python -u main.py > main_0507_1817.log 2>&1 &
tail -f main_0507_1817.log
ps -ef | grep main.py
kill -9 <PID>
"""