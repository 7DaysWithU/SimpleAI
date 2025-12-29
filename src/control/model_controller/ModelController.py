from abc import ABC, abstractmethod
from typing import Callable, Any

from utils import Configurer


class ModelController(ABC, Configurer):
    """
    模型控制类基类
    """

    @abstractmethod
    def load_data(self, dataset: Any) -> None:
        """
        加载数据集和包装数据集

        :param dataset: 原始数据集
        :return: 无

        注意
        ------
        - 本方法会将原始数据集及包装好的训练集和测试集一同以类属性存储
        """

        pass

    @abstractmethod
    def load_model_into_memory(self) -> None:
        """
        加载预训练的模型至内存

        :return: 无
        """

        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """
        获得完整模型的流程, 包括数据集定义, 数据集包装, 模型定义, 模型训练保存

        :return: 无
        """

        pass

    @abstractmethod
    def eval(self) -> tuple[Callable, ...]:
        """
        评估模型的性能, 包括但不限于计算指标、绘制图表

        :return: 绘制函数
        """

        pass

    @abstractmethod
    def use(self, **data: Any) -> Any:
        """
        使用模型完成业务任务

        :param data: 模型所需数据
        :return: 任意
        """

        pass
