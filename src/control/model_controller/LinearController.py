import os
from typing import Callable, Any

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import torch
from torch.optim import Adam
from torch.nn import MSELoss

from control.model_controller import ModelController
from dataset import LinearDataset
from module import LinearRegression
from utils import Loader, Trainer
from common import ModelSaveMode, Checkpoint


class LinearController(ModelController):
    """
    线性回归模型控制类
    """

    def __init__(self):
        """
        初始化参数
        """

        self.model_name = LinearRegression.__name__

        # 设置参数
        self.DEBUG = True
        self.model_dir = '../../../resource/dynamic/saved_models'
        self.model_path = {
            ModelSaveMode.STATE: os.path.join(self.model_dir, f'{self.model_name}_{ModelSaveMode.STATE.value}.pth'),
            ModelSaveMode.FRAME: os.path.join(self.model_dir, f'{self.model_name}_{ModelSaveMode.FRAME.value}.pth')
        }

        # 模型参数
        self.train_ratio = 0.8
        self.batch_size = 100
        self.epoch = 200
        self.incremental_epoch = 1
        self.trials = 5
        self.lr = 1e-2
        self.weight_decay = 0.01

        # 数据集
        self.dataset = None
        self.train_dataloader = None
        self.test_dataloader = None

        # 工具对象
        self.pretrained_model = None

    def configure(self, settings: Any) -> None:
        """
        配置 NCF模型参数

        :param settings: 设置内容
        :return: 无
        """

        # 设置参数
        self.load_configuration(settings['basic'])
        # 模型参数
        self.load_configuration(settings.get('model', {}).get(self.model_name, {}))

        # 修正相对路径
        self.model_dir = self.path_revise(self.model_dir)
        # 在保存路径后补上模型后缀, 作为真正的模型路径
        self.model_path = {
            ModelSaveMode.STATE: os.path.join(self.model_dir, f'{self.model_name}_{ModelSaveMode.STATE.value}.pth'),
            ModelSaveMode.FRAME: os.path.join(self.model_dir, f'{self.model_name}_{ModelSaveMode.FRAME.value}.pth')
        }

    def load_data(self, dataset: LinearDataset) -> None:
        """
        加载原始数据集和包装训练测试集

        :param dataset: 原始数据集
        :return: 无
        """

        self.dataset = dataset

        loader = Loader(dataset)
        self.train_dataloader, self.test_dataloader = loader.get_dataloader(self.train_ratio, self.batch_size)

    def load_model_into_memory(self) -> None:
        """
        加载 NCF模型至内存

        :return: 无
        """

        model = torch.load(self.model_path[ModelSaveMode.FRAME])
        model.eval()

        self.pretrained_model = model

    def train(self, incremental: bool) -> None:
        """
        训练 NCF模型

        :param incremental: 是否增量学习
        :return: 无
        """

        model = LinearRegression(1)
        optimizer = Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        loss = MSELoss()
        if incremental:
            checkpoint = torch.load(self.model_path[ModelSaveMode.STATE])
            # 排除某些层。如在增量训练下, Embedding层会因为输入特征数不匹配而不能加载, 所以要排除
            model.load_state_dict(
                self.checkpoint_filter(
                    model.state_dict(), checkpoint[Checkpoint.MODEL],
                    'embedding'
                )
            )
            # optimizer.load_state_dict(checkpoint[Checkpoint.OPTIMIZER])

        trainer = Trainer(model, optimizer, loss)
        trainer.train(
            supervise = True,
            train_data = self.train_dataloader,
            epoch = self.incremental_epoch if incremental else self.epoch,
            trials = self.trials,
            test_data = self.test_dataloader,
            plot = self.DEBUG,
            model_dir = self.model_dir
        )
        self.load_model_into_memory()

    def eval(self) -> tuple[Callable, ...]:
        """
        评估 NCF模型

        :return: 准确率柱状图绘制函数

        注意
        ------
        - 该评测方法会重新分割数据集, 可检测模型的泛化能力
        """

        def plot_regression_fit() -> None:
            """
            绘制线性回归拟合效果图：真实值 vs 预测值
            适用于单特征线性回归（如 y = w*x + b）

            :return: 无
            """
            # 收集所有测试数据和预测结果
            all_x = []
            all_y_true = []
            all_y_pred = []

            self.pretrained_model.eval()  # 切换到评估模式
            with torch.no_grad():
                for X_batch, y_batch in self.test_dataloader:
                    # X_batch: (batch_size, 1), y_batch: (batch_size,)
                    pred = self.pretrained_model(X_batch).squeeze(1)  # (batch_size,)

                    all_x.append(X_batch.cpu().numpy())
                    all_y_true.append(y_batch.cpu().numpy())
                    all_y_pred.append(pred.cpu().numpy())

            # 合并所有批次
            x_all = np.concatenate(all_x).flatten()
            y_true_all = np.concatenate(all_y_true).flatten()
            y_pred_all = np.concatenate(all_y_pred).flatten()

            # 计算 R² 决定系数（回归版“准确率”）
            r2 = r2_score(y_true_all, y_pred_all)

            # 创建排序索引，以便画出平滑的拟合线
            sorted_idx = np.argsort(x_all)
            x_sorted = x_all[sorted_idx]
            y_pred_sorted = y_pred_all[sorted_idx]

            # 绘图
            plt.figure(figsize = (8, 6))

            # 散点：真实数据
            plt.scatter(x_all, y_true_all, alpha = 0.6, label = 'True values', color = 'skyblue', edgecolor = 'black')

            # 拟合线：模型预测（按 x 排序后连线）
            plt.plot(x_sorted, y_pred_sorted, color = 'red', linewidth = 2, label = f'Predicted (R² = {r2:.3f})')

            plt.title(f"{self.model_name} Fit")
            plt.xlabel('Input feature (x)')
            plt.ylabel('Target (y)')
            plt.legend()
            plt.grid(True, linestyle = '--', alpha = 0.5)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            plt.show()

        return (plot_regression_fit,)

    def use(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测指定X对应的y

        :param X: 要预测的值
        :return: X对应的值
        """

        return self.pretrained_model(X)
