import os
import sys
from abc import abstractmethod
from collections import OrderedDict
from typing import Any


class Configurer:
    """
    协助完成设置配置

    使用
    ------
    - 将本类作为父类继承, 在子类中使用 load_configure方法加载配置集中的配置
    >>> class MyCls(Configurer):
    >>>     def __init__(self):
    >>>         self.attribute1 = None
    >>>         self.attribute2 = None
    >>>
    >>>     def configure(self, settings: Any)-> None:
    >>>         self.load_configuration(settings)
    >>>
    >>> my_settings = {
    >>>     "attribute1": "value1",
    >>>     "attribute2": "value2"
    >>> }
    >>>
    >>> my_cls = MyCls()
    >>> my_cls.configure(my_settings)

    - 修正项目中的相对路径, 使其自适应运行环境
    >>> relative_path = 'data.csv'
    >>> absolute_path = Configurer.path_revise(relative_path)

    - 加载状态字典
    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>>
    >>> class MyModel(nn.Module, Configurer):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>
    >>>         self.layer1 = nn.Linear(1, 1)
    >>>         self.layer2 = nn.Embedding(1, 1)
    >>>
    >>>     def configure(self, settings: Any) -> None:
    >>>         pass
    >>>
    >>>
    >>> model = MyModel()
    >>> checkpoint = torch.load('my_model.pth')
    >>> model.load_state_dict(
    >>>     MyModel.checkpoint_filter(model.state_dict(), checkpoint, 'layer1') #type:ignore
    >>> )
    """

    @abstractmethod
    def configure(self, settings: dict[str, Any] | str) -> None:
        """
        加载预设参数

        :param settings: 配置内容
        :return: 无
        """

        pass

    def load_configuration(self, settings: dict[str, Any]) -> None:
        """
        从指定配置集中加载实现类已有的类属性的配置

        :return: 无

        注意
        ------
        - 该方法仅加载已有的类属性, 不会加载实现类中未定义的属性, 即使该属性出现在设置集中
        - 该方法为具体实现类的 configure方法的分割实现, 主要作用为加载多个不同设置集时保持程序的易维护性, 建议在configure方法中使用此方法
        """

        for attr in settings:
            if hasattr(self, attr):
                value = settings[attr]
                if value is not None:
                    setattr(self, attr, value)

    @classmethod
    def path_revise(cls, relative_path: str) -> str:
        """
        修正相对路径以匹配不同的运行环境

        :param relative_path: 相对路径
        :return: 对应运行环境下的绝对路径
        """

        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            module = cls.__module__
            base_path = os.path.dirname(os.path.abspath(sys.modules[module].__file__))

        return os.path.join(base_path, relative_path)

    @staticmethod
    def checkpoint_filter(model_state_dict: OrderedDict | dict, pretrained_state_dict: OrderedDict,
                          *exclusions: str) -> OrderedDict:
        """
        加载状态字典

        :param model_state_dict: 模型的状态字典
        :param pretrained_state_dict: 预训练的状态字段
        :param exclusions: 排除层
        :return: 排除了指定层的模型状态字典
        """

        for layer in list(pretrained_state_dict.keys()):
            if any(excluded_layer in layer for excluded_layer in exclusions):
                del pretrained_state_dict[layer]

        model_state_dict.update(pretrained_state_dict)
        return model_state_dict
