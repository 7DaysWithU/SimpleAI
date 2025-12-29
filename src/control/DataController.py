from typing import Any

from database import MySQL, Redis, Faiss, LocalData
from dataset import LinearDataset
from control.dataset_controller import LinearDatasetController
from utils import Configurer


class DataController(Configurer):
    """
    数据控制类, 负责数据的优化存取
    """

    def __init__(self):
        """
        初始化参数
        """

        # 请求轮询时间(s)
        self.poll_time = 1.0

        # self.sql = MySQL()
        # self.redis = Redis()
        # self.faiss = Faiss()
        self.local_data = LocalData()
        self.databases = [
            # self.sql,
            # self.redis,
            # self.faiss,
            self.local_data
        ]

        self.linear_dataset_controller = LinearDatasetController()

    def configure(self, settings: dict[str, Any]) -> None:
        """
        配置数据库及缓存设置

        :param settings: 设置内容
        :return: 无
        """

        for database in self.databases:
            database.configure(settings)

    def connect(self) -> None:
        """
        连接所有数据库

        :return: 无
        """

        for database in self.databases:
            database.connect()

    def close(self) -> None:
        """
        关闭所有数据库

        :return: 无
        """

        for database in self.databases:
            database.close()

    def get_LinearDataset(self) -> LinearDataset:
        """
        获取 LinearDataset

        :return: LinearDataset 数据集
        """

        original_data = self.local_data.get_linear_data()
        self.linear_dataset_controller.data = original_data

        return self.linear_dataset_controller.get_dataset()
