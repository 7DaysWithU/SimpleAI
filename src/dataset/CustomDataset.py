from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import Dataset
import copy


class CustomDataset(ABC, Dataset):
    """
    自定义数据集类, 在 Dataset的基础上扩展更多操作
    """

    def __init__(self, *args, **kwargs):
        """
        预设数据

        :param args: 位置参数
        :param kwargs: 关键字参数

        注意
        ------
        - 采用列表更方便后期根据需求扩展成 ndarray或 tensor
        """

        self.data = np.array(None)

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
    def __getitem__(self, item: int) -> None:
        """
        获得数据集指定下标样本

        :param item: 下标
        :return: 对应下标的数据样本

        注意
        ------
        - 基类中作抽象方法, 请在每个子类中具体实现
        """

        pass

    @staticmethod
    def train_test_split(dataset: 'CustomDataset', train_ratio: float) -> tuple['CustomDataset', 'CustomDataset']:
        """
        将数据集分为训练和测试两部分

        :param dataset: 继承了 CustomDataset的子类数据集
        :param train_ratio: 训练样本比例
        :return: 训练数据集, 测试数据集
        """

        train_dataset = copy.deepcopy(dataset)
        test_dataset = copy.deepcopy(dataset)

        original_data = dataset.data.copy()
        np.random.shuffle(original_data)

        split_index = int(train_ratio * len(original_data))
        train_dataset.data = original_data[:split_index]
        test_dataset.data = original_data[split_index:]

        return train_dataset, test_dataset
