import torch
import pandas as pd

from dataset import CustomDataset


class LinearDataset(CustomDataset):
    """
    线性数据集
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        读入原始数据

        :param dataframe: 数据DataFrame
        """

        super().__init__()

        self.data = dataframe.to_numpy()

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获得单个样本

        :param item: 样本标号
        :return: 一维特征张量, 一维标签张量
        """

        return torch.tensor(self.data[item][:-1], dtype = torch.float32), torch.tensor(self.data[item][-1:], dtype = torch.float32)
