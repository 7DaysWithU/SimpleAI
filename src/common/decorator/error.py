class OverrideError(Exception):
    """
    重写异常: 父类中不存在该重写方法

    使用
    ------
    - error_code: 0
    >>> method_name = 'my_method'
    >>> raise OverrideError(0, method = method_name)

    - error_code: 1
    >>> cls_name = 'MyCls'
    >>> method_name = 'my_method'
    >>> raise OverrideError(1, cls_name, method_name)

    - error_code: 2
    >>> cls_name = 'MyCls'
    >>> method_name = 'my_method'
    >>> raise OverrideError(2, cls = cls_name)

    注意
    ------
    以下为重写异常详细分类

    - error_code: 0 要重写的方法在基类中不存在
    - error_code: 1 实现类未实现要重写的方法
    - error_code: 2 实现类为抽象基类
    """

    def __init__(self, error_code: int, cls: str = None, method: str = None):
        """
        获得触发异常的方法名

        :param error_code: 具体异常代码
        :param cls: 触发异常的类
        :param method: 触发异常的方法
        """

        self.error_code = error_code
        self.cls = cls
        self.method = method

    def __str__(self) -> str:
        """
        自定义异常输出信息

        :return: 异常输出信息
        """

        error_code_mapping = {0: f"No such method called '{self.method}' in base classes",
                              1: f"No such method called '{self.method}' in class '{self.cls}'",
                              2: f"Class '{self.cls}' as abstract class cannot override methods"}

        return error_code_mapping[self.error_code]
