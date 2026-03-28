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

    parser.add_argument('-bs', '--batch_size', type = int, default = 32)

    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('-emb', '--embedding_dim', type=int, default=512)
    parser.add_argument('--emb_align_way', type=str, default='interpolate')

    parser.add_argument('-eptr', '--epochs_train', type=int, default = 300)#
    parser.add_argument('-eptu', '--epochs_tune', type=int, default = 200)#
    parser.add_argument('-lrp', '--learning_rate_pretrain', type=float, default = 5e-2)
    parser.add_argument('-lrt', '--learning_rate_tune', type=float, default = 5e-3)
    parser.add_argument('-mod', '--model_name', type=str, default = "CMTarget")
    parser.add_argument('--model_path', type = str, default="")

    parser.add_argument('--patience', type = int, default=10) 
    parser.add_argument('-pTok', '--protein_encoder_Token_num', type=int, default=416)
    # parser.add_argument('-scD', '--score_emb_dim', type = int, default = 256)
    parser.add_argument('-scW', '--score_way', type=str, default='MF', 
                        help="choose a scorer, MF,GMF,Cosine ")
    

    parser.add_argument('--source_name', type = str, default="drugbank")
    parser.add_argument('--target_name', default='hit')
    parser.add_argument('--task', type=str, default = "train", 
                        help="the stage:train, predict")
    # parser.add_argument('--timestamp', type=str, default = "001")



    args = parser.parse_args()

    config = {}
    config['batch_size'] = args.batch_size
    config['checkpoint_interval'] = args.checkpoint_interval
    config['emb'] = args.embedding_dim
    config['emb_align_way'] = args.emb_align_way
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

    if configs['task'] == 'train':
        print(f"⚡train model {configs['model']}: epoch_pretrain: {configs['epochs_train']},\
               epochs_tune: {configs['epochs_tune']}, batch_size: {configs['batch_size']}, \
               pretrain-lr:{configs['pretrain_learning_rate']},tune-lr: {configs['tune_learning_rate']}")

        trainer = CMTargetTrainer(configs, configs['source_name'], configs['model_path'])
        trainer.train(pretrain_output_path)
        
        fineTunner = FineTunner(configs, configs['target_name'], configs['model_path'])#model
        fineTunner.fineTune(fintune_output_path)    # 加载pre_train完毕后的model_path, 作为初始值
        

    elif configs['task'] == 'predict':
        if not os.path.exists(configs['model_path']):
            print("please make sure the configs['model_path'] is exist." \
            "If it is none, please execute the training phase")
            sys.exit()
        
        predictor = CMTargetPredictor(configs, configs['model_path'])#model
        predictor.predict()


"""
nohup /opt/conda/envs/cmtarget1/bin/python feature_save_hf.py /root/gpufree-data/workplace/CMTarget-LLM/ > feature_df_03170915.log 2>&1 &
tail -f feature_df_03170915.log
ps -ef | grep feature_save_hf.py
nvidia-smi
kill -9 <PID>

nohup python -u main.py > main_0317_1300.log 2>&1 &
tail -f main_0317_1300.log
ps -ef | grep main.py
"""