from abc import abstractmethod
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    自定义数据集类, 在 Dataset的基础上扩展更多操作
    """

    def __init__(self, data: pd.DataFrame | np.ndarray, *args, **kwargs):
        """
        预设数据

        :param data: 原始数据.
                    支持pandas.DataFrame, 以便后续扩展操作
                    支持numpy.ndarray, 用于__len__和train_test_split方法
        :param args: 位置参数
        :param kwargs: 关键字参数
        """

        self.dataframe = pd.DataFrame(None)
        self.data = np.array(None)

        if isinstance(data, pd.DataFrame):
            self.dataframe = data
            self.data = data.to_numpy()
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError("data must be DataFrame or ndarray")

    def __len__(self) -> int:
        """
        获得数据集长度

        :return: 数据集长度

        注意
        ------
        - 此基类方法返回的是最外层列表的元素个数
        - 如有特殊长度处理需求请在子类中重写 __len__方法
        """

        return len(self.data)

    @abstractmethod
    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        获得数据集指定下标样本

        :param item: 下标
        :return: 对应下标的数据样本

        注意
        ------
        - 基类中作抽象方法, 请在每个子类中具体实现
        """

        pass

    def train_valid_test_split(
            self,
            train_ratio: float,
            valid_ratio: float
    ) -> tuple['CustomDataset', 'CustomDataset', 'CustomDataset']:
        """
        将数据集分为训练和测试两部分

        :param train_ratio: 训练集比例
        :param valid_ratio: 验证集比例
        :return: 训练数据集, 测试数据集
        """

        original_data = self.data.copy()
        np.random.shuffle(original_data)

        split_index_train = int(train_ratio * len(original_data))
        split_index_valid = int(valid_ratio * len(original_data))
        train_dataset = original_data[:split_index_train]
        valid_dataset = original_data[split_index_train:split_index_train + split_index_valid]
        test_dataset = original_data[split_index_train + split_index_valid:]

        return type(self)(train_dataset), type(self)(valid_dataset), type(self)(test_dataset)
