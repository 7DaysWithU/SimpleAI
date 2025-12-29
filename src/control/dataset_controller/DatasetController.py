from abc import ABC, abstractmethod

from dataset import CustomDataset


class DatasetController(ABC):
    """
    数据集控制类基类
    """

    @abstractmethod
    def get_dataset(self, *args, **kwargs) -> CustomDataset:
        """
        获得数据集

        :param args: 位置参数
        :param kwargs: 关键词参数
        :return: 数据集
        """

        pass
