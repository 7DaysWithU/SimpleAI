import os
import pandas as pd
import numpy as np


class Discretization:
    """
    字段离散化处理器

    使用
    ------
    - 离散化指定数据

    >>> original_data = ['F', 'M', 'F', 'F']
    >>> discretizer = Discretization()
    >>> discretizer.discretize(original_data, 'gender')

    - 获得离散化任务的数据

    >>> original_data = ['F', 'M', 'F', 'F']
    >>> discretizer = Discretization()
    >>> discretizer.discretize(original_data, 'gender')
    >>> discretized_gender = discretizer.get_discretized_mapping('gender')
    """

    def __init__(self, file_dir: str):
        """
        初始化参数

        :param file_dir: 离散化任务处理结果的存放路径
        """

        self.file_dir = file_dir

    def discretize(self, data: list[int | str], task: str) -> None:
        """
        将给定数据进行离散化处理

        :param data: 原始数据
        :param task: 任务键, 后续可根据该键对单个同任务数据离散化
        :return: 无
        """

        unique = set(data)
        discretization = np.array([[value, idx] for idx, value in enumerate(unique)])

        discretized_df = pd.DataFrame({
            'original': discretization[:, 0],
            'discretized': discretization[:, 1]
        })
        discretized_df.to_csv(os.path.join(self.file_dir, f'{task}.csv'), index = False, encoding = 'UTF-8')

    def get_discretized_mapping(self, task: str) -> pd.DataFrame:
        """
        获得指定任务的离散化数据

        :param task: 给定的任务
        :return: 离散化数据
        """

        return pd.read_csv(os.path.join(self.file_dir, f'{task}.csv'))
