import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    """
    线性回归模型
    """

    def __init__(self, in_features: int):
        """
        初始化参数

        :param in_features: 输入特征向量维度
        """

        super().__init__()
        self.output_layer = nn.Linear(in_features, 1, dtype = torch.float32)

    def forward(self, X: torch.Tensor) -> float:
        """
        前向传播

        :param X: 特征向量
        :return: 预测向量
        """

        a = self.output_layer(X)

        return a
