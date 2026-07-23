import torch
import torch.nn as nn
from torch.optim import Optimizer
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os.path
from tqdm import tqdm
from typing import Any

from .Stopper import Stopper
from common import ModelSaveMode, Checkpoint


class CTRTrainer:
    """
    训练 CTR 排序模型
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, loss: nn.Module, device: torch.device):
        """
        初始化训练参数

        :param model: 训练模型
        :param optimizer: 优化器
        :param loss: 损失器
        """

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device

    def train(self,
              train_data: DataLoader,
              epoch: int,
              trials: int = None,
              valid_data: DataLoader = None,
              plot: bool = None,
              model_dir: str = None,
              check_per_step: int = None) -> None:
        """
        训练模型

        :param train_data: 训练数据集
        :param epoch: 迭代次数
        :param trials: 允许的最大迭代次数
        :param valid_data: 测试数据集
        :param plot: 是否绘制损失图像
        :param model_dir: 模型保存路径
        :param check_per_step: 检查步长
        :return: 无

        注意
        ------
        - 若要使用早停辅助, 请指定 trials和 test_data
        - 若要保存模型, 请指定 model_dir
        """

        model_name = self.model.__class__.__name__

        stopper = Stopper(trials) if trials and valid_data else None
        train_loss_history = []
        val_loss_history = []

        # 迭代训练
        for e in tqdm(range(1, epoch + 1), desc = f'Training {model_name}'):
            self.model.train()

            # 一次完整数据集训练
            total_loss = 0
            for x_dict, y in train_data:
                if y.dim() == 1:
                    y = y.view(-1, 1)
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)

                self.optimizer.zero_grad()
                model_output = self.model(x_dict)
                iteration_loss = self.loss(model_output, y)
                iteration_loss.backward()
                self.optimizer.step()

                total_loss += iteration_loss.item()

            # 统计本次迭代平均损失
            average_loss = total_loss / len(train_data.dataset)  # type: ignore
            train_loss_history.append(average_loss)

            # 早停
            if stopper:
                valid_loss = self.evaluate(valid_data)
                val_loss_history.append(valid_loss)
                if stopper.can_stop(valid_loss):
                    break

            # 检查点. 最后一个epoch不执行, 因为最后有总体保存
            if check_per_step and e % check_per_step == 0 and e != epoch:
                self.save_and_show(plot, train_loss_history, val_loss_history, model_dir, model_name)

        # 总体保存
        self.save_and_show(plot, train_loss_history, val_loss_history, model_dir, model_name)

    def evaluate(self, eval_data: DataLoader) -> float:
        """
        评估训练结果

        :param eval_data: 验证数据集 或 测试数据集
        :return: 平均单例损失
        """

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x_dict, y in eval_data:
                if y.dim() == 1:
                    y = y.view(-1, 1)
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)

                model_output = self.model(x_dict)
                test_loss = self.loss(model_output, y)
                total_loss += test_loss.item()

        average_loss = total_loss / len(eval_data.dataset)  # type: ignore
        return average_loss

    def evaluate_auc(self, eval_data: DataLoader) -> float:
        """
        验证准确率

        :param eval_data: 验证数据集 或 测试数据集
        :return: 准确率
        """

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x_dict, y in eval_data:
                if y.dim() == 1:
                    y = y.view(-1, 1)
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)

                pred = self.model(x_dict).cpu().numpy().flatten()
                all_preds.extend(pred)
                all_labels.extend(y.cpu().numpy().flatten())
        return roc_auc_score(all_labels, all_preds)

    def save_and_show(self,
                      plot: bool = False,
                      train_loss_history = None,
                      valid_loss_history = None,
                      model_dir: str = False,
                      model_name: str = None) -> None:
        """
        检查损失并保存状态

        :param plot: 是否绘制损失图像
        :param train_loss_history: 训练损失
        :param valid_loss_history: 验证损失
        :param model_dir: 模型保存路径
        :param model_name: 模型名称
        :return: 无
        """

        if model_dir:
            self.save(model_dir, model_name)

        if plot:
            self.show(train_loss_history, valid_loss_history)

    def show(self, *args: Any) -> None:
        """
        绘制损失图

        :param args: 损失记录列表
        :return: 无
        """

        labels = ['Train Loss', 'Valid Loss']

        plt.figure(figsize = (8, 8))
        for idx, x in enumerate(args):
            plt.plot(x, label = labels[idx])

        plt.title(f'Loss during the training process of {self.model.__class__.__name__}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.legend()
        # plt.savefig(f'Z:/信息/计算机设计大赛/我的/2025/素材/test/{self.model.__class__.__name__}_loss.svg', dpi = 1000, bbox_inches = 'tight')
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
