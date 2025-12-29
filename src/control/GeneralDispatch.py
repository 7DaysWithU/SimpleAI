import gc
import json
import time

import torch

from control import DataController
from control.model_controller import LinearController
from utils import Configurer


class GeneralDispatch(Configurer):
    """
    对所有控制类做总调度, 计算最终推荐结果
    """

    def __init__(self):
        """
        初始化参数
        """

        # 数据库重连时间(秒)
        self.db_expire_time = 600
        # 数据库最后连接时间
        self.last_connected_timestamp = 0

        self.model1_controller = LinearController()
        # self.model2_controller = Model2()
        # self.model3_controller = Model3()

        self.data_controller = DataController()
        self.model_controllers = [
            self.model1_controller,
            # self.model2_controller,
            # self.model3_controller
        ]

    def __enter__(self) -> None:
        """
        非 连接各个数据库

        :return: 无
        """

        now_timestamp = int(time.time())
        if now_timestamp - self.last_connected_timestamp >= self.db_expire_time:
            self.data_controller.connect()
            self.last_connected_timestamp = now_timestamp

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        非 下断开各个数据库连接

        :param exc_type: with块中发生异常时异常的类型
        :param exc_val: with块中发生异常时异常的值
        :param exc_tb: with块中发生异常时异常的堆栈信息
        :return: 无
        """

        self.data_controller.close()

    def configure(self, config_path: str) -> None:
        """
        加载模型控制类和数据控制类参数

        :param config_path: 配置文件地址
        :return: 无

        注意
        ------
        - 使用该配置方法将会对所有模型及控制类按 settings.json内的设置进行配置, 若不使用该方法则全部模型及控制类使用默认配置
        """

        with open(config_path, 'r', encoding = 'UTF-8') as file:
            settings = json.load(file)

        self.data_controller.configure(settings)
        for model_controller in self.model_controllers:
            model_controller.configure(settings)

    def load_model(self) -> None:
        """
        将模型及索引载入内存

        :return: 无
        """

        for model_controller in self.model_controllers:
            model_controller.load_model_into_memory()

    def train(self, incremental: bool) -> None:
        """
        批量更新所有模型

        :param incremental: 是否增量学习
        :return: 无
        """

        for idx, model_controller in enumerate(self.model_controllers):
            # 加载数据集
            linear_dataset = self.data_controller.get_LinearDataset()
            model_controller.load_data(linear_dataset)

            # 训练模型
            model_controller.train(incremental = incremental)

            # 清理内存
            gc.collect()

    def eval(self) -> None:
        """
        评估所有模型

        :return: 无
        """

        eval_funcs = []
        for model_controller in self.model_controllers:
            eval_funcs.extend(model_controller.eval())

        for eval_func in eval_funcs:
            eval_func()

    def use(self, X: list[float]) -> torch.Tensor:
        """
        预测指定X对应的y

        :param X: 要预测的值
        :return: X对应的值
        """

        return self.model1_controller.use(torch.tensor([X]))
