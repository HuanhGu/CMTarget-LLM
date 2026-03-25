## 3.17待办，给pretrain_test添加loss分析

## 问题：
![alt text](9aeac0382cee143e1f74c9c1c9a6e9b.png)
## 关于MOE
看百度MOE
https://github.com/huggingface/transformers/blob/c368e139aade3ee7cdfa29387f3249168a912e5c/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py#L355


## 关于输入的特征对齐方式
4. 更好的替代方案？如果你觉得插值法效果不佳，可以考虑以下更符合 Transformer 逻辑的方法：

Adaptive Average Pooling (自适应平均池化)：torch.nn.AdaptiveAvgPool1d(target_size)。这在保留全局信息方面通常比单纯的线性插值更鲁棒。
注意：需要先将维度转置为 $[B, C, N]$。

Learnable Resampling (可学习重采样)：引入一个固定长度的 Query（如一组 Latent Tokens），通过 Cross-Attention 去原序列中提取信息。这是 Perceiver IO 等模型的核心思想。

Global Attention Pooling：通过一个线性层计算每个 Token 的权重，加权求和得到固定维度的表示，而不是硬性缩放。

总结建议如果你的 $N$ 维度代表的是空间或时间上的连续信号（如图像块、语音帧），你可以放心使用 interpolate。但如果 $N$ 维度是语言单词，建议优先使用 Padding/Truncate（补齐/截断）。如果必须缩放，请确保在插值后检查模型是否还能正确识别序列的边界。



## 任务待办

✅❌

| 任务待办                           |      |
| ---------------------------------- | ---- |
| ❌把predictor写一下                 |      |
| ❌把特征提取encoder放到dataloader里 |      |
|                                    |      |



```
1.把特征提取放到dataloader，模型调用时get_dataloader()，就能直接调用特征向量。
2.模型调用时dataloader时，只能得到文本序列，训练时encoder成特征向量
从性能和封装的角度聊聊哪个更好
```

**重复计算：** 如果训练 100 个 Epoch，方案 B 就会把同样的文本重复编码 100 次，这在算力资源紧张时是巨大的浪费。

建议一：对于“静态特征”，选方案 A

如果你的特征提取逻辑是**不可学习的**（例如：TF-IDF 向量、手动提取的统计特征、或者已经冻结参数的预训练模型输出），请务必放在 DataLoader 中甚至直接离线存入数据库/硬盘。

建议二：对于“可学习特征”，选方案 B

如果特征提取部分（如 Embedding 层、CNN/RNN Encoder）需要随损失函数**同步更新参数**，则必须放在模型内部。



## github链接

https://github.com/HuanhGu/MyCMTarget

## 消融试验：

chemberta 的cls_embedding 使用第一层的embedding，还是last_state



## dataloader时的问题：

第一个batch：
torch.Size([3, 2, 416, 100]) 
torch.Size([3, 2, 43, 768])

第二个batch
torch.Size([3, 1, 501, 100]) 
torch.Size([3, 1, 68, 768])

```bash
Traceback (most recent call last):
  File "d:\Workplace\MyCMTarget\CMTarget-llm\main.py", line 81, in <module>
    encoder_hit.encode_and_save()
  File "d:\Workplace\MyCMTarget\CMTarget-llm\data_process.py", line 100, in encode_and_save
    all_protein_features = torch.cat(all_protein_features, dim=1)   # 变为tensor类型
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 501 but got size 416 for tensor number 1 in the list.
```



主函数调用dataloader直接encoder所有数据然后存储下来：【失败】

- 因为每个蛋白质序列的token数目不同，使用pad方法将所有batch的蛋白质token长度对齐，感觉 不太好。
- 最终：直接在训练时encoder,浪费点时间也就算了【在CMTargetTrainer中添加encoder】

dustbin代码：

```python
  # 2. 读取序列数据,创建dataloader
        # 序列文件路径
    drugbank_path = Path("./data/dataset/drugbank/drugbank.csv")
    hit_path = Path("./data/dataset/hit/hit.csv")
        # encoder-dataloader 保存路径
    encoder_savepath_hit = Path("./data/encoder/encoded_hit.pt")
    encoder_savepath_drugbank = Path("./data/encoder/encoded_drugbank.pt")

    # 序列encoder + 得到encoder_loader + 保存到.pt
    if not encoder_savepath_hit.exists():
        print("hit_encoder_loader文件不存在")
        hit_dataset = DTIDataset(hit_path)
        hit_dataloader = DataLoader(hit_dataset, batch_size=2, shuffle=True, num_workers=0) # 序列的loader
        encoder_hit = SequencePreEncoder(hit_dataloader, encoder_savepath_hit) # 所有的序列编码
        encoder_hit.encode_and_save()
    else:
        print("编码文件已存在，跳过编码")

    encoded_dataset = EncodedDTIDataset(encoder_savepath_hit)   # 从文件路径加载encoder_loader
    train_loader = DataLoader(encoded_dataset, batch_size=2,shuffle=True) # encoder_loader
 

    '''
        drugbank_dataset = DTIDataset(drugbank_path)
        drugbank_dataloader = DataLoader(drugbank_dataset, batch_size=2, shuffle=True, num_workers=0)
        encoder_drugbank = SequencePreEncoder(drugbank_dataloader, encoder_savepath_drugbank)   ###
    

    if not encoder_savepath_drugbank.exists():
        print("drugbank_encoder_loader文件不存在")
        encoder_drugbank.encode_and_save()
    '''
```

## 调参试验——对比损失模块

对比损失时应该计算样本级余弦相似度还是token级余弦相似度

```python
        # 计算样本级余弦相似度    
        z_i_pool = z_i.mean(dim=1)   # [B, D]
        z_j_pool = z_j.mean(dim=1)   # [B, D]
        sim_matrix = F.cosine_similarity(z_i_pool.unsqueeze(1), z_j_pool.unsqueeze(0), dim=-1) #(batch_size, batch_size)
        # 计算token级余弦相似度  
        # sim_matrix = F.cosine_similarity(
        #     z_i.unsqueeze(2),   # [B, T, 1, D]
        #     z_j.unsqueeze(1),   # [B, 1, T, D]
        #     dim=-1
        # )  #[2, 501, 501]
```

- 答案：计算样本级余弦相似度才对，应为后边要和样本的labels比较，得到bce损失的~



##  负载均衡损失原理



负载均衡损失：$L_{moe}=E * ∑_{i=1}^E f_i^2$

其中：

- E 是专家数量
- $f_i$ 是第 i 个专家的平均分配概率



希望：每个专家被选中的概率尽可能平均。理想状况下，$f_i = 1/E$，此时Loss=1，为最小值。





## BCE损失



outputs：

(tensor([0.4859, 0.4870], grad_fn=<SigmoidBackward0>), tensor(604.9968, grad_fn=<AddBackward0>), tensor(70.1427, grad_fn=<AddBackward0>))

label：

tensor([[1.],
        [0.]])





## Encoder、Embdding、Forward、Inference区别

- Encoder：序列→token_id

- Embdding：token_id→连续向量
- Forward：每个 `nn.Module` 都必须有def forward(self, x):
- Inference（推理）：模型不更新参数、不反向传播、只做预测，就叫推理。





```
def train(self):
        train_dataloader, test_dataloader = self.dataloader
        max_f1 = 0
        
        for i in range(self.epochs):
            loss = self.train_model(self.model, train_dataloader, i)
```








## 其它
时间戳

```python
from datetime import datetime

# 生成格式：年 月 日 _ 时 分 秒
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config['timestamp'] = timestamp
print(timestamp)  # 输出示例: 20260304_180521
```





```python
config['batch_size'] = args.batch_size
config['emb'] = args.embedding_dim
config['epochs'] = args.epochs


config['learning_rate'] = args.learning_rate

config['model'] = args.model_name
config['model_path'] = args.model_path

config['score_way'] = args.score_way
# config['score_dim'] = args.score_emb_dim
config['source'] = args.source
config['target'] = args.target
config['task'] = args.task

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
config['timestamp'] = timestamp

# config['timestamp'] = args.timestamp
config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

