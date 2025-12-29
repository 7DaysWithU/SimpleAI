import pandas as pd

from control.dataset_controller import DatasetController
from dataset import LinearDataset


class LinearDatasetController(DatasetController):
    """
    LinearDataset控制类
    """

    def __init__(self):
        self.__data = None

    @property
    def data(self) -> pd.DataFrame:
        """
        data getter方法

        :return: self.__data
        """

        return self.__data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        """
        data setter 方法

        :param data: 需要设置的data
        :return: 无
        """

        self.__data = data

    def get_dataset(self) -> LinearDataset:
        """
        获得 LinearDataset 数据集

        :return: LinearDataset 数据集
        """

        return LinearDataset(self.data)
