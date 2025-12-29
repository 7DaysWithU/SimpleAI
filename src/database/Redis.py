import redis
import socket
import subprocess
import json
from typing import Any

from database import Database


class Redis(Database):
    """
    Redis内存数据库
    """

    def __init__(self):
        """
        初始化参数
        """

        self.database_name = self.__class__.__name__

        self.redis_server_exe = 'your_path/Redis/redis-server.exe'

        self.host = 'localhost'
        self.port = 6379
        self.db = 0
        self.expire_seconds = 60

        self.process = None
        self.client = None

    def configure(self, settings: dict[str, Any]) -> None:
        """
        配置缓存设置

        :param settings: 设置内容
        :return: 无
        """

        self.load_configuration(settings.get('database', {}).get(self.database_name, {}))

    def connect(self) -> None:
        """
        连接到 Redis客户端

        :return: 无

        注意
        ------
        - 只有是由本类开启 Redis服务端的情况下, 最终可以选择关闭服务端。使用连接池获得连接时仅能释放连接, 而无法关闭服务端
        """

        # 启动Redis客户端
        if not self.is_redis_server_running():
            self.process = subprocess.Popen(self.redis_server_exe)

        self.client = redis.StrictRedis(host = self.host, port = self.port, db = self.db, decode_responses = False)

    def close(self) -> None:
        """
        关闭 Redis客户端

        :return: 无

        注意
        ------
        - 仅当由本类开启服务端的情况下调用此方法方可关闭服务端, 非本类启动服务器的情况下调用此方法无效
        """

        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def is_redis_server_running(self) -> bool:
        """
        检查 Redis服务端是否正在运行

        :return: Redis服务端是否正在运行
        """

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect((self.host, self.port))
            s.close()
            return True
        except (socket.error, ConnectionRefusedError):
            return False

    def set(self, key: Any, value: Any) -> None:
        """
        设置键值对

        :param key: 键
        :param value: 值
        :return: 无
        """

        # 完全存储方法: 序列化
        # serialized_value = pickle.dumps(value)
        # compressed_value = zlib.compress(serialized_value)
        # self.client.set(key, compressed_value, ex = self.expire_seconds)

        json_value = json.dumps(value)
        self.client.set(key, json_value, ex = self.expire_seconds)

    def get(self, key: Any) -> Any:
        """
        获取指定键的值

        :param key: 键
        :return: 值或 None
        """

        # 完全存储方法: 序列化
        # compressed_value = self.client.get(key)
        # decompressed_value = zlib.decompress(compressed_value)
        # return pickle.loads(decompressed_value)

        json_value = self.client.get(key)
        return json.loads(json_value)

    def delete(self, key: Any) -> None:
        """
        删除指定键

        :param key: 键
        :return: 值或 None
        """

        self.client.delete(key)

    def exists(self, key: Any) -> bool:
        """
        检查键是否存在

        :param key: 键
        :return: True 键是否存在
        """

        return self.client.exists(key)

    def setnx(self, key: Any, expire_time: int) -> bool:
        """
        获得Redis锁

        :param key: 锁键
        :param expire_time: 锁失效时间(s)
        :return: 是否成功获得锁
        """

        return self.client.set(key, 1, nx = True, ex = expire_time)
