
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        '''
        eps是为了防止除以0的情况
        这个是防止除以0的参数 epsilon ε
        '''
        self.eps = eps
        '''
        weight是RMSNorm的可学习参数，形状初始化为dim，每个元素初始化为1
        这意味着在没有训练的情况下，RMSNorm的行为类似于标准化
        这个是要学的参数γ
        '''
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        '''        
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        # '''
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        '''        
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight γ ，这是RMSNorm的一个可学习的缩放因子
        # '''
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
