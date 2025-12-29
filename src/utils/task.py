import time
import uuid
import traceback

from dataclasses import dataclass
import typing
from typing import Callable, Any, Protocol, ParamSpec, TypeVar
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from common import TaskState

P = ParamSpec('P')
R = TypeVar('R')


class LongTask(Protocol[P, R]):
    """
    长任务鸭子类型
    """

    # 是否为长任务
    _is_long_task: bool

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        函数鸭子类型匹配

        :param args: 原函数参数
        :param kwargs: 原函数参数
        :return: 原函数返回值类型
        """

        ...


@dataclass
class Task:
    """
    任务类
    """

    # 任务id
    task_id: str
    # 任务状态
    state: TaskState
    # 任务信息
    info: Any
    # 任务函数
    task_future: Future

    def pending(self) -> None:
        """
        进入等待状态

        :return: 无
        """

        self.state = TaskState.PENDING

    def started(self) -> None:
        """
        进入开始状态

        :return: 无
        """

        self.state = TaskState.STARTED

    def sent(self) -> None:
        """
        任务已被发送至任务队列

        :return: 无
        """

        self.state = TaskState.SENT

    def progress(self) -> None:
        """
        任务正在执行

        :return: 无
        """

        self.state = TaskState.PROGRESS

    def retry(self) -> None:
        """
        任务重试

        :return: 无
        """

        self.state = TaskState.RETRY

    def revoked(self) -> None:
        """
        撤销任务

        :return: 无
        """

        self.state = TaskState.REVOKED

    def success(self) -> None:
        """
        任务成功执行

        :return: 无
        """

        self.state = TaskState.SUCCESS

    def failure(self) -> None:
        """
        任务执行失败

        :return: 无
        """

        self.state = TaskState.FAILURE

    def set_info(self, info: Any) -> None:
        """
        设置任务信息

        :param info: 任务信息
        :return: 无
        """

        self.info = info


class TaskManager:
    """
    异步任务管理类
    """

    def __init__(self, max_workers: int):
        """
        初始化参数

        :param max_workers: 异步任务队列最大子进程数
        """

        self.__tasks: dict[str, Task] = {}
        self.__process_pool = ThreadPoolExecutor(max_workers = max_workers)
        self.__process_lock = threading.Lock()
        self.__tasks_status_monitor = threading.Thread(target = self.__monitor, daemon = True)
        self.__tasks_status_monitor.start()

    def __str__(self):
        """
        显式任务队列

        :return: 任务队列字符串
        """

        s = 'Task Queue: {\n'
        for task_id, task in self.__tasks.items():
            s += f"\t{str(task)}\n"
        s += '}'

        return s

    @staticmethod
    def long_task(func: Callable[P, R]) -> LongTask[P, R]:
        """
        标记函数为长任务

        :param func: 被标记函数
        :return: 标记后的新函数
        """

        func._is_long_task = True
        return typing.cast(LongTask[P, R], func)

    def submit(self, func: LongTask, *args: Any, **kwargs: Any) -> str:
        """
        注册任务

        :param func: 被注册函数
        :return: 包装函数地址
        """

        # 拒绝将未注册LongTask的任务加入异步任务队列
        if not getattr(func, '_is_long_task', False):
            raise TypeError(f"The target task '{func.__name__}' is not a long task, please use '@TaskManager.long_task' to decorate it")

        task_id = str(uuid.uuid4())
        task_future = self.__process_pool.submit(func, *args, **kwargs)
        with self.__process_lock:
            self.__tasks.update({task_id: Task(task_id, TaskState.PENDING, None, task_future)})

        return task_id

    def __monitor(self) -> None:
        """
        监控任务队列, 自动修改任务状态

        :return: 无
        """

        while True:
            time.sleep(0.1)

            with self.__process_lock:
                for task_id, task in list(self.__tasks.items()):
                    task_future = task.task_future

                    # 查看任务状态
                    if task_future.running():
                        task.progress()
                    elif task_future.done():
                        try:
                            task_future.result()
                        except Exception:
                            task.failure()
                            task.set_info(str(traceback.format_exc()))
                        else:
                            task.success()

    def result(self, task_id: str) -> Task | None:
        """
        获得指定 id的任务

        :param task_id: 任务 id
        :return: 任务
        """

        # 获取注册任务及其线程
        with self.__process_lock:
            task = self.__tasks.get(task_id, None)
        if task is None:
            raise ValueError(f"No such task called '{task_id}'")

        # 最终结果一定为成功或失败, 任务执行完毕后, 若外部获取任务状态则视为对任务状态的ACK, 删除任务队列中的任务
        if task.state is TaskState.SUCCESS or task.state is TaskState.FAILURE:  # type: ignore
            self.__tasks.pop(task_id)

        return task
