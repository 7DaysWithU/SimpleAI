from enum import Enum


class ConnectionMode(Enum):
    """
    数据库连接模式
    """

    # 使用本地数据文件, 不启用SQL, 不启用Redis
    LOCAL_ONLY = 'LOCAL_ONLY'
    # 使用本地数据文件, 不启用SQL, 启用Redis
    LOCAL_REDIS = 'LOCAL_REDIS'
    # 启用SQL, 不启用Redis
    SQL_ONLY = 'SQL_ONLY'
    # 启用SQL, 启用Redis
    SQL_REDIS = 'SQL_REDIS'


class Addition(Enum):
    """
    用户-物品交互附加项
    """

    # 无附加信息
    NONE = 'NONE'
    # 附加用户信息
    USER = 'USER'
    # 附加除主键外的全部信息
    FULL = 'FULL'


class SamplingFormat(Enum):
    """
    数据集采样格式
    """

    # 逐点的(一般仅包含正样本)
    POINTWISE = 'PointWise'
    # 逐对的(一般包含一个正样本和一个负样本)
    PAIRWISE = 'PairWise'
    # 逐列表的(一般包含一个正样本和多个负样本)
    LISTWISE = 'ListWise'


class ModelSaveMode(Enum):
    """
    模型保存方式
    """

    # 保存完整模型
    FRAME = 'FRAME'
    # 保存模型参数
    STATE = 'STATE'


class Checkpoint(Enum):
    """
    模型状态字典内容
    """

    # 模型的状态字典
    MODEL = 'MODEL'
    # 优化器的状态字典
    OPTIMIZER = 'OPTIMIZER'


class TaskState(Enum):
    """
    任务状态
    """

    # 任务等待中
    PENDING = 202, 0
    # 任务已开始
    STARTED = 202, 1
    # 任务已发送到队列
    SENT = 202, 2
    # 任务进度更新
    PROGRESS = 200, 3
    # 任务重试
    RETRY = 503, 4
    # 任务已撤销
    REVOKED = 403, 5
    # 任务成功完成
    SUCCESS = 200, 6
    # 任务执行失败
    FAILURE = 500, 7
