from torch.utils.data import DataLoader

from dataset import CustomDataset


class Loader:
    """
    加载数据集为DataLoader
    """

    def __init__(self, dataset: CustomDataset):
        """
        初始化参数

        :param dataset: 继承了Dataset类的数据集
        """

        self.dataset = dataset

    def get_dataloader(self, train_ratio: float, batch_size: int) -> tuple[DataLoader, DataLoader]:
        """
        获得训练dataloader和测试dataloader

        :param train_ratio: 训练集占比
        :param batch_size: 批大小
        :return: 训练集dataloader, 测试集dataloader
        """

        # 分割原始数据集为train和test
        train_dataset, test_dataset = self.dataset.train_test_split(self.dataset, train_ratio)

        # 封装加载为DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = False)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, drop_last = False)

        return train_dataloader, test_dataloader
