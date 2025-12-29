import faiss
import torch
import numpy as np
from typing import Any

from database import Database


class Faiss(Database):
    """
    Faiss向量数据库
    """

    def __init__(self):
        """
        初始化参数
        """

        self.database_name = self.__class__.__name__

        self.local_index_path = {
            "index_1": "../../resource/dynamic/index_1.faiss",
            "index_2": "../../resource/dynamic/index_2.faiss"
        }

    def configure(self, settings: dict[str, Any]) -> None:
        """
        配置 Faiss数据库

        :param settings: 设置内容
        :return: 无
        """

        self.load_configuration(settings.get('database', {}).get(self.database_name, {}))

        for index in self.local_index_path:
            self.local_index_path[index] = self.path_revise(self.local_index_path[index])

    def connect(self) -> None:
        """
        连接 Faiss数据库

        :return: 无
        """

        pass

    def close(self) -> None:
        """
        关闭 Faiss数据库

        :return: 无
        """

        pass

    @staticmethod
    def tensor_to_ndarray(tensor: torch.Tensor) -> np.ndarray:
        """
        将 tensor转化为 ndarray

        :param tensor: 待转化的 tensor
        :return: 转化后的 ndarray
        """

        return tensor.detach().numpy().astype('float32')

    @staticmethod
    def ndarray_to_tensor(ndarray: np.ndarray) -> torch.Tensor:
        """
        将 ndarray转化为 tensor

        :param ndarray: 待转化的 ndarray
        :return: 转化后的 tensor
        """

        return torch.tensor(ndarray)
