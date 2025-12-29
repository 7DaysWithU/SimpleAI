from abc import ABC, abstractmethod

from utils import Configurer


class Database(ABC, Configurer):
    """
    数据库抽象基类
    """

    @abstractmethod
    def connect(self) -> None:
        """
        连接数据库

        :return: 无
        """

        pass

    @abstractmethod
    def close(self) -> None:
        """
        关闭连接

        :return: 无
        """

        pass
