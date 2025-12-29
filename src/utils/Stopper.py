class Stopper:
    """
    实现训练早停机制
    """

    def __init__(self, trials: int):
        """
        初始化参数

        :param trials: 允许的最大迭代次数
        """

        self.trials = trials
        self.now_trials = 0
        self.best_loss = 1e9

    def can_stop(self, current_loss: float) -> bool:
        """
        判断是否可以停止训练

        :param current_loss: 当前损失
        :return: 是否可以停止训练
        """

        # 损失更好: 继续训练尝试获得更优损失
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.now_trials = int(self.now_trials * 0.8)
            return False

        # 损失未达到历史最优损失且在允许训练的最大次数内: 继续训练尝试获得更优损失
        elif self.now_trials + 1 < self.trials:
            self.now_trials += 1
            return False

        # 既未创新最优损失, 也超出了允许的最大迭代次数: 停止训练
        else:
            return True
