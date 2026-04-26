
# MOE介绍及代码： https://blog.csdn.net/weixin_44986037/article/details/150105895
# QwenMoe&DeepSeekMoe : https://techdiylife.github.io/blog/blog.html?category1=c02&blogid=0074

from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class Qwen2MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["swish"]# 可选

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(self, dim, expert_dim, expert_number):
        super().__init__()
        self.num_experts = expert_number
        self.top_k = 2

        # 专家路由 : 选一个 expert 
        self.gate = nn.Sequential(nn.Linear(dim, expert_number, bias=False))
        self.experts = nn.ModuleList([Qwen2MoeMLP(dim, expert_dim) for _ in range(expert_number)])

        # 与Mixtral相比，Qwen2-MoE多了 shared_expert 和 shared_expert_gate
        self.shared_expert = Qwen2MoeMLP(dim, intermediate_size=expert_dim)
        self.shared_expert_gate = torch.nn.Linear(dim, 1, bias=False)

        # self.output_norm = nn.LayerNorm(dim, expert_dim)
    

    
    def forward(self, hidden_states, att_mask) -> torch.Tensor:  
        "输入: 融合特征 [128, 506, 512]"
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # A. 权重
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) 
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # if self.norm_topk_prob:
            # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 专家掩码：只使用特定专家
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # 遍历专家 计算
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # 专家 * 自己的权重
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # 共享专家 代码合并处理部分（此处与DeepSeek不一样）
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        aux_moe_loss = load_balancing_loss_func(
            router_logits,
            self.num_experts,            # 专家总数
            self.top_k,    # 每个 token 分配的专家数
            att_mask,              # mask 掩码，忽略 padding token
        )
        
        return final_hidden_states, aux_moe_loss




#=============================================================
class BasicMOE(nn.Module):
    def __init__(self, dim, expert_dim, expert_number):
        super().__init__()
        self.experts = nn.ModuleList([DeepseekV2MLP(dim, expert_dim) for _ in range(expert_number)])
        # MOEDTA
        # self.experts = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(dim, expert_dim),
        #         SwiGLU(expert_dim, expert_dim),  
        #         # nn.ReLU(),  
        #         nn.Linear(expert_dim, expert_dim),
        #         nn.Dropout(p=0.1)
        #     ) for _ in range(expert_number)])

        self.expert_number = expert_number
        self.top_k = 2
        
        # 专家路由 : 选一个 expert 
        self.gate = nn.Sequential(
            nn.Linear(dim, expert_number, bias=False) # 简单即正义
        )
        # self.softmax = nn.Softmax(dim=-1)
        self.output_norm = nn.LayerNorm(dim, expert_dim)
    


    def forward(self, x):  
        "输入: 融合特征 [128, 506, 512]"
        # A. 权重
        routing_weights = F.softmax(self.gate(x), dim=-1)  # (batch, token_num, expert_number) 
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        f = routing_weights.view(-1, self.expert_number).mean(0) # [501, 3] 专家权重
        moe_loss = self.expert_number * torch.sum(f * f) #负载均衡损失 Loss = N * sum(f_i * P_i)
        
        # B. 获得所有专家输出
        expert_output = torch.stack([expert(x) for expert in self.experts], dim=2) #[128,506,3, 512]

        # C. 加权求和
        routing_weights = routing_weights.unsqueeze(-1) # (2,501,3,1) * (2, 501, 3, 256) 
        output = torch.sum(routing_weights * expert_output, dim=2) # (2, 501, 256)
        # output = self.output_norm(output)
        
        return output, moe_loss




# 手动为每个专家添加适配器（Adapter）的伪代码
class AdapterExpert(nn.Module):
    def __init__(self, original_expert):
        super().__init__()
        self.original_expert = original_expert
        # 冻结原始专家
        for p in self.original_expert.parameters():
            p.requires_grad = False
        # 添加旁路 LoRA 层 (A, B)
        self.lora_a = nn.Linear(in_dim, r)
        self.lora_b = nn.Linear(r, out_dim)

    def forward(self, x):
        return self.original_expert(x) + self.lora_b(self.lora_a(x))
    


    
class BasicExpert(nn.Module):
    """
    基础专家网络

    输入:
        x : 输入的特征向量(batch, feature_in)
    
    参数:
        feature_in : 输入特征向量的维度
        feature_out : 输出特征向量的维度, 也是Linear的嵌入维度emd_dim
    
    输出:
        单个专家处理后的输出向量
    """
    # 一个 Expert 可以是一个最简单的， linear 层即可
    # 也可以是 MLP 层
    # 也可以是 更复杂的 MLP 层（active function 设置为 swiglu）
    def __init__(self, feature_in, feature_out):
        super().__init__()
        # self.linear = nn.Linear(feature_in, feature_out)
        # 这里使用典型的 MLP 结构：Linear -> ReLU -> Linear
        self.net = nn.Sequential(
            nn.Linear(feature_in, feature_in * 2),
            nn.ReLU(),
            nn.Linear(feature_in * 2, feature_out)
        )
    
    def forward(self, x):
        # return self.linear(x)
        return self.net(x)
    



class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["swish"]#可选

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out)
        self.w2 = nn.Linear(dim_in, dim_out)
        self.w3 = nn.Linear(dim_out, dim_out)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.w3(self.swish(self.w1(x)) * self.w2(x))
    

'''
import torch
import torch.nn as nn

## --- 1. 准备模型 ---
# 假设 model 是你已经初始化好的、包含 AdapterExpert 的大模型
model = YourMainModel() 

# 加载预训练权重（这是微调的前提）
model.load_state_dict(torch.load("pretrained_model.pth"))

## --- 2. 设置微调开关 (核心分隔区) ---
def ready_for_finetuning(model):
    # 冻结所有
    for param in model.parameters():
        param.requires_grad = False
        
    # 只放开每个专家里的 LoRA 层和路由层 (Router)
    # 因为路由层决定了怎么分配给这些带 Adapter 的专家
    for name, module in model.named_modules():
        if "lora_" in name or "router" in name:
            for param in module.parameters():
                param.requires_grad = True

ready_for_finetuning(model)

## --- 3. 定义优化器 ---
# 这里只传入 requires_grad=True 的参数
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
criterion = nn.MSELoss() # 假设是回归任务

## --- 4. 训练循环 (微调过程) ---
model.train() 
for epoch in range(10):
    for batch in dataloader:
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播 (此时 original_expert 的权重参与计算，但不产生梯度)
        output = model(batch.inputs)
        loss = criterion(output, batch.labels)
        
        # 反向传播 (只有 lora_a, lora_b 和 router 会更新)
        loss.backward()
        optimizer.step()

# 保存时，其实只需要保存 LoRA 的权重（可以极大地节省空间）
torch.save({k: v for k, v in model.state_dict().items() if "lora" in k}, "lora_adapter.pth")


'''

"""
def test_basic_moe():
    x = torch.rand(2, 4)

    basic_moe = BasicMOE(4, 3, 2)
    out = basic_moe(x)
    print(out)
    print("out.shape:", out.shape)  #(batch, feature_out), from (2, 4) to (2,3) 


test_basic_moe()

"""