import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import os.path
from tqdm import tqdm
from typing import Any

from .Stopper import Stopper
from common import ModelSaveMode, Checkpoint


class PairwiseTrainer:
    """
    训练 Pairwise 排序模型
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
            for x_dict, _ in train_data:
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}

                self.optimizer.zero_grad()
                pos_scores, neg_scores = self.model(x_dict)
                loss = self.loss(pos_scores, neg_scores)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # 统计本次迭代平均损失
            avg_loss = total_loss / len(train_data.dataset)  # type: ignore
            train_loss_history.append(avg_loss)

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
            for x_dict, _ in eval_data:
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}

                pos_scores, neg_scores = self.model(x_dict)
                loss = self.loss(pos_scores, neg_scores)
                total_loss += loss.item()

        average_loss = total_loss / len(eval_data.dataset)  # type: ignore
        return average_loss

    def evaluate_pairwise_accuracy(self, eval_data: DataLoader) -> float:
        """
        计算每个用户的 Pairwise Accuracy(正样本评分 > 负样本评分的比例), 然后取平均

        :param eval_data: 验证数据集或测试数据集(三元组格式)
        :return: 准确率(0~1)
        """

        self.model.eval()
        user_correct = {}
        user_total = {}

        with torch.no_grad():
            for x_dict, _ in eval_data:
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}

                pos_scores, neg_scores = self.model(x_dict)
                user_ids = x_dict['user_id'].cpu().numpy()
                for i, uid in enumerate(user_ids):
                    if uid not in user_correct:
                        user_correct[uid] = 0
                        user_total[uid] = 0
                    # 每组三元组产生一个比较对
                    user_correct[uid] += (pos_scores[i] > neg_scores[i]).item()  # type: ignore
                    user_total[uid] += 1

        # 计算每个用户的正确率，再平均
        acc_per_user = [user_correct[uid] / user_total[uid] for uid in user_correct]
        return sum(acc_per_user) / len(acc_per_user) if acc_per_user else 0.0

    def evaluate_auc(self, eval_data: DataLoader) -> float:
        """
        计算用户平均 ROC AUC(需要将三元组展开为正负样本对)

        :param eval_data: 验证数据集或测试数据集(三元组格式)
        :return: AUC 值(0~1)
        """

        self.model.eval()
        user_aucs = []
        user_scores = {}

        with torch.no_grad():
            for x_dict, _ in eval_data:
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}

                pos_scores, neg_scores = self.model(x_dict)
                user_ids = x_dict['user_id'].cpu().numpy()
                for idx, user_id in enumerate(user_ids):
                    if user_id not in user_scores:
                        user_scores[user_id] = {'pos': [], 'neg': []}
                    user_scores[user_id]['pos'].append(pos_scores[idx].item())
                    user_scores[user_id]['neg'].append(neg_scores[idx].item())

        for user_id, scores in user_scores.items():
            y_true = [1] * len(scores['pos']) + [0] * len(scores['neg'])
            y_score = scores['pos'] + scores['neg']
            if len(set(y_true)) == 2:  # 正负都存在
                auc = roc_auc_score(y_true, y_score)
                user_aucs.append(auc)

        return sum(user_aucs) / len(user_aucs) if user_aucs else 0.0

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
