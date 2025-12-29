import mysql.connector
import pandas as pd
import json
from typing import Any

from database import Database


class MySQL(Database):
    """
    MySQL关系数据库
    """

    def __init__(self):
        """
        初始化参数
        """

        self.database_name = self.__class__.__name__

        # 数据库IP地址
        self.host: str = 'localhost'
        # 数据库端口
        self.port: int = 3306
        # 数据库用户名
        self.user: str = 'root'
        # 数据库密码
        self.password: str = 'your_password'
        # 数据库名称
        self.database: str = 'your_database'

        # 数据库表命名
        self.table = {
            "table_1": {
                "table_name": "table_1",
                "col_name": {
                    "logical_col_1": "physical_col_1",
                    "logical_col_2": "physical_col_2",
                }
            },
            "table_2": {
                "table_name": "table_2",
                "col_name": {
                    "logical_col_1": "physical_col_1",
                    "logical_col_2": "physical_col_2",
                }
            }
        }
        # 数据库视图命名
        self.view = {
            'view_1': {
                'view_name': 'view_1',
                'col_name': {
                    "logical_col_1": "physical_col_1",
                    "logical_col_2": "physical_col_2",
                }
            }
        }

        self.conn = None
        self.cursor = None

    def configure(self, settings: dict[str, Any]) -> None:
        """
        配置数据库及缓存设置

        :param settings: 设置内容
        :return: 无
        """

        self.load_configuration(settings.get('database', {}).get(self.database_name, {}))

    def connect(self) -> None:
        """
        连接数据库

        :return: 无
        """

        self.conn = mysql.connector.connect(host = self.host, port = self.port, user = self.user,
                                            password = self.password, database = self.database)
        self.cursor = self.conn.cursor()

    def close(self) -> None:
        """
        关闭数据库

        :return: 无
        """

        self.cursor.close()
        self.conn.close()
