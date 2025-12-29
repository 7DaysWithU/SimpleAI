from threading import Lock
from typing import Any

from common.decorator import singleton


@singleton
class Persistencer:
    """
    运行时在内存中持久化实例
    """

    def __init__(self):
        """
        初始化参数
        """

        self.__instances = {}
        self.__lock = Lock()

    def __str__(self) -> str:
        """
        返回维护信息

        :return: 维护信息
        """

        info = [f'self id: {id(self)}']

        for key, instance in self.__instances.items():
            key_instance = (f"key: {key}\t\t"
                            f"instance: "
                            f"(type) {type(instance)}\t"
                            f"( id ) {id(instance)}")
            info.append(key_instance)

        return '\n'.join(info)

    def update(self, **key_instance: Any) -> None:
        """
        增加维护的实例

        :param key_instance: 任意数量的 key-instance对
        :return: 无
        """

        with self.__lock:
            self.__instances.update(key_instance)

    def remove(self, *key: str) -> None:
        """
        去除指定的维护实例

        :param key: 实例索引键
        :return: 无
        """

        with self.__lock:
            for each_key in key:
                del self.__instances[each_key]

    def get(self, key: str) -> Any:
        """
        获得指定的维护实例

        :param key: 实例索引键
        :return: 指定的维护实例
        """

        return self.__instances.get(key, None)

    def exist(self, key: str) -> bool:
        """
        检查是否存在指定键

        :param key: 实例索引键
        :return: 是否存在该指定键
        """

        return key in self.__instances
