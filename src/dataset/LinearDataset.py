import torch
import pandas as pd

from dataset import CustomDataset


class LinearDataset(CustomDataset):
    """
    线性数据集
    """

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        获得单个样本

        :param item: 样本标号
        :return: 一维特征张量, 一维标签张量
        """

        features = {
            'x': torch.tensor(self.data[item][:-1], dtype = torch.float32)
        }
        label = torch.tensor(self.data[item][-1:], dtype = torch.float32)
        return features, label
