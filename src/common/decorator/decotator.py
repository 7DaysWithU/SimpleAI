import functools
from datetime import datetime
import inspect
import logging
import traceback
import threading
import time
from typing import Callable, Type, Any
from abc import ABC

from .error import OverrideError


def timer(log_path: str | None = None) -> Callable:
    """
    装饰器: 统计函数运行信息

    :param log_path: 运行信息输出日志路径
    :return: 接收函数的地址

    使用
    ------
    - 位置
    函数定义处

    >>> @timer()
    >>> def my_function():
    >>>     pass

    类方法定义处

    >>> class MyClass:
    >>>     @timer()
    >>>     def class_method(self):
    >>>         pass

    - 模式
    日志输出

    >>> @timer(log_path = './log.txt')
    >>> def my_function():
    >>>     pass

    控制台输出

    >>> @timer()
    >>> def my_function():
    >>>     pass

    注意
    ------
    - 此装饰器提供日志输出和控制台输出, 生产模式下请选择日志输出
    """

    def receiver(func: Callable) -> Callable:
        """
        接收被装饰函数

        :param func: 被装饰函数
        :return: 触发函数地址
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            计算函数运行时间和打印函数运行信息的包装函数

            :param args: 被装饰函数的不定长位置参数
            :param kwargs: 被装饰函数的不定长关键字参数
            :return: 被装饰函数的返回值
            """

            module_dir = inspect.getsourcefile(func)

            time_start = {'seconds': time.time(), 'dates': datetime.now()}
            result = func(*args, **kwargs)
            time_end = {'seconds': time.time(), 'dates': datetime.now()}

            logging.basicConfig(filename = log_path, level = logging.INFO,
                                format = '[%(asctime)s][%(name)s][%(levelname)s]\n'
                                         '%(message)s')
            logger = logging.getLogger()
            log_message = '\n'.join((f"Program:\t{func.__qualname__}",
                                     f"Source:\t\t{module_dir}",
                                     f"Start time:\t{time_start['dates']}",
                                     f"End time:\t{time_end['dates']}",
                                     f"Spend time:\t{(time_end['seconds'] - time_start['seconds']):.6f} s"))
            logger.info(msg = log_message)

            return result

        return wrapper

    return receiver


def singleton(cls: Type) -> Callable:
    """
    装饰器: 线程安全地实现类单例模式

    :param cls: 要实现单例模式的类
    :return: 触发函数地址

    使用
    ------
    - 将该装饰器放置在类定义处
    >>> @singleton
    >>> class MyClass:
    >>>     pass

    注意
    ------
    - 使用该装饰器的类仅可被实例化一次. 若强行多次实例化, 后续得到的实例对象将始终为第一次实例化时的对象
    """

    instance = {}
    lock = threading.Lock()

    def wrapper(*args: Any, **kwargs: Any) -> Callable:
        """
        在线程安全的情况下实现类单例模式

        :param args: 被装饰类的不定长位置参数
        :param kwargs: 被装饰类的不定长关键字参数
        :return: 被装饰类的返回值
        """

        with lock:
            if cls not in instance:
                instance[cls] = cls(*args, **kwargs)
            return instance[cls]

    return wrapper


def override(methods: list[str]):
    """
    装饰器: 标注重写方法并检测其重写合法性

    :param methods: 要重写的方法
    :return: 接收函数地址

    使用
    ------
    >>> class A:
    >>>     def hello(self):
    >>>         pass
    >>>
    >>>
    >>> @override(['hello'])
    >>> class B(A):
    >>>     def hello(self):
    >>>         pass

    注意
    ------
    - 若被装饰方法不存在于任何基类中, 将触发OverrideError异常
    - 若被装饰方法不存在于实现类中, 将触发OverrideError异常
    - 若实现类为抽象基类, 将触发OverrideError异常
    """

    def receiver(cls: Type) -> Callable:
        """
        接受被装饰类

        :param cls: 要重写方法所属的类
        :return: 触发函数地址
        """

        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            """
            判别该方法是否是父类中存在的方法

            :param args: 被装饰方法的不定长位置参数
            :param kwargs: 被装饰方法的不定长关键字参数
            :return: 被装饰方法的返回值
            """

            # 检测实现类是否为抽象基类
            if issubclass(cls, ABC):
                raise OverrideError(2, cls.__name__)

            # 检查基类中是否存在实现类中要重写的方法
            bases = cls.__bases__
            for method in methods:
                method_in_bases = any((hasattr(base, method) and callable(getattr(base, method)) for base in bases))
                if not method_in_bases:
                    raise OverrideError(0, method = method)

            # 检查实现类中是否全部实现要重写的方法
            local_methods = inspect.getmembers(cls, predicate = lambda member: (inspect.isfunction(member) or
                                                                                inspect.ismethod(member)))
            local_methods_name = [method[0] for method in local_methods]
            if not set(methods).issubset(set(local_methods_name)):
                error_method_name = list(set(methods) - set(local_methods_name))[0]
                raise OverrideError(1, cls.__name__, error_method_name)

            return cls(*args, **kwargs)

        return wrapper

    return receiver


def deprecated(alternative: str = None, log_path: str | None = None) -> Callable:
    """
    装饰器: 标记弃用函数

    :param alternative: 可替代使用的函数
    :param log_path: 运行信息输出日志路径
    :return: 接收函数的地址

    使用
    ------
    >>> class MyClass:
    >>>     @deprecated('func_b')
    >>>     def func_a(self):
    >>>         pass
    >>>
    >>>     def func_b(self):
    >>>         pass

    注意
    ------
    - 此装饰器提供日志输出和控制台输出, 生产模式下请选择日志输出
    """

    def receiver(func: Callable) -> Callable:
        """
        接收被装饰函数

        :param func: 被装饰函数
        :return: 触发函数地址
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            标记并提示函数已被弃用

            :param args: 被装饰函数的不定长位置参数
            :param kwargs: 被装饰函数的不定长关键字参数
            :return: 被装饰函数的返回值
            """

            result = func(*args, **kwargs)

            logging.basicConfig(filename = log_path, level = logging.WARNING,
                                format = '[%(asctime)s][%(name)s][%(levelname)s]\n'
                                         '%(message)s')
            logger = logging.getLogger()

            stack_info = traceback.format_stack()
            stack_trace = ''.join(stack_info)

            # 记录警告和堆栈追踪信息
            message = ((f"{stack_trace}"
                        f"DeprecationWarning: '{func.__qualname__}' has deprecated and will be removed in future versions. ") +
                       (f"Please use an alternative." if alternative is None else
                        f"Please use '{alternative}'."))
            logger.warning(message)

            return result

        return wrapper

    return receiver
