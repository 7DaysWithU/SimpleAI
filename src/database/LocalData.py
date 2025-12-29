import pandas as pd
from typing import Any

from database import Database


class LocalData(Database):
    """
    本地数据文件查取
    """

    def __init__(self):
        """
        初始化参数
        """

        self.database_name = self.__class__.__name__

        self.local_data_file_path = {
            "file_1": "../resource/static/file_1.csv",
            "file_2": "../resource/static/file_2.csv",
            "linear_data": "../resource/static/linear_data.csv"
        }

    def configure(self, settings: dict[str, Any]) -> None:
        """
        配置本地数据文件路径

        :param settings: 设置内容
        :return: 无
        """

        self.load_configuration(settings.get('database', {}).get(self.database_name, {}))

        # 将相对路径变为绝对路径, 以免运行时出错
        for file in self.local_data_file_path:
            self.local_data_file_path[file] = self.path_revise(self.local_data_file_path[file])

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_linear_data(self) -> pd.DataFrame:
        """
        读取数据文件并返回数据的 DataFrame

        :return: 数据的 DataFrame
        """

        dataframe = pd.read_csv(self.local_data_file_path['linear_data'])
        return dataframe
