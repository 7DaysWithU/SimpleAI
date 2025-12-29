import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os.path
from tqdm import tqdm
from typing import Any

from .Stopper import Stopper
from common import ModelSaveMode, Checkpoint


class Trainer:
    """
    训练并评估各类继承了nn.Module类的模型
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, loss: nn.Module):
        """
        初始化训练参数

        :param model: 训练模型
        :param optimizer: 优化器
        :param loss: 损失器
        """

        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def train(self,
              supervise: bool,
              train_data: DataLoader,
              epoch: int,
              trials: int = None,
              test_data: DataLoader = None,
              plot: bool = None,
              model_dir: str = None,
              check_rate: float = None) -> None:
        """
        训练模型

        :param supervise: 是否使用标签监督学习
        :param train_data: 训练数据集
        :param epoch: 迭代次数
        :param trials: 允许的最大迭代次数
        :param test_data: 测试数据集
        :param plot: 是否绘制损失图像
        :param model_dir: 模型保存路径
        :param check_rate: 检查点频率
        :return: 无

        注意
        ------
        - 若要使用早停辅助, 请指定 trials和 test_data
        - 若要保存模型, 请指定 model_dir
        """

        model_name = self.model.__class__.__name__
        train_loss_history = []
        test_loss_history = []
        stopper = Stopper(trials) if trials else None
        checking_epoch = None if check_rate is None else int(check_rate * epoch)

        # 迭代训练
        for e in tqdm(range(1, epoch + 1), desc = f'Training {model_name}'):
            self.model.train()

            # 一次完整数据集训练
            total_loss = 0
            for data in train_data:
                train_X, train_y = data[:-1], data[-1]

                self.optimizer.zero_grad()
                model_output = self.model(*train_X)
                iteration_loss = self.calculate_loss(supervise, model_output, train_y)
                iteration_loss.backward()
                self.optimizer.step()

                total_loss += iteration_loss.item()

            # 统计本次迭代平均损失
            average_loss = total_loss / len(train_data.dataset)  # type: ignore
            train_loss_history.append(average_loss)

            # 早停
            if trials:
                test_loss = self.evaluate(supervise, test_data)
                test_loss_history.append(test_loss)
                if stopper.can_stop(test_loss):
                    break

            # 检查点
            if checking_epoch and e % checking_epoch == 0:
                self.save_and_show(plot, train_loss_history, test_loss_history, model_dir, model_name)

        # 总体保存
        self.save_and_show(plot, train_loss_history, test_loss_history, model_dir, model_name)

    def calculate_loss(self,
                       supervise: bool,
                       model_output: torch.Tensor | tuple[torch.Tensor, ...],
                       label: torch.Tensor = None) -> torch.Tensor:
        """
        计算 batch内损失

        :param supervise: 是否为监督学习
        :param model_output: 模型输出
        :param label: 数据集标签
        :return: batch内损失
        """

        multi_output = isinstance(model_output, tuple)

        loss_mapping = {
            (True, True): lambda: self.loss(*model_output, label),
            (True, False): lambda: self.loss(model_output, label),
            (False, True): lambda: self.loss(*model_output),
            (False, False): lambda: self.loss(model_output),
        }

        return loss_mapping[(supervise, multi_output)]()

    def evaluate(self, supervise: bool, test_data: DataLoader) -> float:
        """
        评估训练结果

        :param supervise: 是否使用标签监督学习
        :param test_data: 测试数据集
        :return: 平均单例损失
        """

        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            for data in test_data:
                test_X, test_y = data[:-1], data[-1]
                model_output = self.model(*test_X)
                test_loss = self.calculate_loss(supervise, model_output, test_y)
                total_loss += test_loss

        average_loss = total_loss / len(test_data.dataset)  # type: ignore

        return average_loss

    def save_and_show(self,
                      plot: bool = False,
                      train_loss_history = None,
                      test_loss_history = None,
                      model_dir: str = False,
                      model_name: str = None) -> None:
        """
        检查损失并保存状态

        :param plot: 是否绘制损失图像
        :param train_loss_history: 训练损失
        :param test_loss_history: 测试损失
        :param model_dir: 模型保存路径
        :param model_name: 模型名称
        :return: 无
        """

        if model_dir:
            self.save(model_dir, model_name)

        if plot:
            self.show(train_loss_history, test_loss_history)

    def show(self, *args: Any) -> None:
        """
        绘制损失图

        :param args: 损失记录列表
        :return: 无
        """

        labels = ['Train loss', 'Valid loss']

        plt.figure(figsize = (8, 8))
        for idx, x in enumerate(args):
            plt.plot(x, label = labels[idx])

        plt.title(f'Loss during the training process of {self.model.__class__.__name__}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.legend()
        # plt.savefig(f'{self.model.__class__.__name__}_loss.svg', dpi = 1000, bbox_inches = 'tight')
        plt.show()

    def save(self, model_dir: str, model_name: str) -> None:
        """
        保存模型

        :param model_dir: 模型保存路径
        :param model_name: 模型名称
        :return: 无
        """

        # 保存整体模型
        torch.save(self.model, os.path.join(model_dir, f'{model_name}_{ModelSaveMode.FRAME.value}.pth'))
        # 保存模型参数
        checkpoint = {
            Checkpoint.MODEL: self.model.state_dict(),
            Checkpoint.OPTIMIZER: self.optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(model_dir, f'{model_name}_{ModelSaveMode.STATE.value}.pth'))
