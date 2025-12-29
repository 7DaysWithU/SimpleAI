import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer


class TextureAnalysis:
    """
    此类用作分割给定文本获得标签

    使用
    ------
    - 提取文本关键词
    使用默认tag_number

    >>> text = '今天天气真好！'
    >>> your_stopwords_filepath = 'stopwords_file.txt' # Use your actual path
    >>> analyzer = TextureAnalysis(your_stopwords_filepath)
    >>> tags = analyzer.analysis(text, mode = 'tag')

    手动指定tag_number

    >>> text = '今天天气真好！'
    >>> your_stopwords_filepath = 'stopwords_file.txt' # Use your actual path
    >>> analyzer = TextureAnalysis(your_stopwords_filepath)
    >>> tags = analyzer.analysis(text, mode = 'tag', tag_number = 15)

    - 提取文本词频表
    >>> text = '今天天气真好'
    >>> your_stopwords_filepath = 'stopwords_file.txt' # Use your actual path
    >>> analyzer = TextureAnalysis(your_stopwords_filepath)
    >>> words_frequency = analyzer.analysis(text, mode = 'frequency')

    - 提取文本TF-IDF权重表
    >>> text = '今天天气真好'
    >>> your_stopwords_filepath = 'stopwords_file.txt' # Use your actual path
    >>> analyzer = TextureAnalysis(your_stopwords_filepath)
    >>> words_weight = analyzer.analysis(text, mode = 'weight')

    - 更换实例停用词库
    >>> your_stopwords_filepath = 'stopwords_file.txt' # Use your actual path
    >>> analyzer = TextureAnalysis(your_stopwords_filepath)
    >>> new_stopwords_filepath = "new_stopwords_file.txt"
    >>> analyzer.set_stopwords(new_stopwords_filepath)

    注意
    ------
    - 实例不会将待分析文本作为类属性, 因此建议本类实行单例模式
    """

    def __init__(self, stopwords_filepath: str):
        """
        导入预设好的停用词库

        :param stopwords_filepath: 停用词文件路径

        注意
        ------
        - 实例在创建时将会自动设置一次停用词库
        """

        self.stopwords = set()
        self.set_stopwords(stopwords_filepath)

    def analysis(self, text: str, mode: str, tag_number: int = 10) -> list[str] | dict[str, int] | dict[str, float]:
        """
        自主选择分析模式获取对应的分析结果

        :param text: 要处理的原始文本
        :param mode: 选择的处理方式
        :param tag_number: tag模式下提取的关键词数量
        :return: 关键词列表 或 词频字典 或 TF-IDF权重字典

        使用
        ------
        - mode = ['tag', 'frequency', 'weight']

        注意
        -------
        - 请使用对应的模式来完成分析功能
        - tag模式默认tag_number = 10, 若需要修改请使用关键字参数
        """

        cleaned_wordlist = self.clean_text(text)

        mode_mapping = {'tag': lambda: self.get_tags(cleaned_wordlist, tag_number),
                        'frequency': lambda: self.get_words_frequency(cleaned_wordlist),
                        'weight': lambda: self.get_words_weight(cleaned_wordlist)}
        if mode not in mode_mapping:
            raise ValueError(f"Invalid keyword '{mode}'. Expected one of: {list(mode_mapping.keys())}")

        return mode_mapping[mode]()

    def set_stopwords(self, stopwords_filepath: str) -> None:
        """
        将停用词库导入

        :param stopwords_filepath: 停用词文件路径
        :return: 无

        使用
        ------
        - 给定预先准备好的停用词文件路径, 本方法会导入该文件内的停用词作为该实例的停用词库

        注意
        ------
        - 实例在创建时将会自动设置一次停用词库
        """

        # 放入集合
        with open(stopwords_filepath, 'r', encoding = 'UTF-8') as file:
            for word in file:
                self.stopwords.add(word.strip())

    def clean_text(self, original_text: str) -> list[str]:
        """
        去除原始文本中的停用词

        :param original_text: 未经处理的原始文本
        :return: 文本按词切分并去除停用词后的词列表

        使用
        ------
        - 将原始文本以字符串的形式作为本方法的输入参数, 清洗后获得文本的词汇表
        """

        # 原始文本切分为词
        words = jieba.lcut(original_text)
        cleaned_text = []

        # 去除停用词
        for word in words:
            allow_add = True
            for stopword in self.stopwords:
                if word == stopword or word in stopword or stopword in word:
                    allow_add = False
                    break
            if allow_add:
                cleaned_text.append(word)

        return cleaned_text

    def get_tags(self, cleaned_wordlist: list[str], tag_number: int) -> list[str]:
        """
        获得给定文本的标签

        :param cleaned_wordlist: 处理后的词汇表
        :param tag_number: 要获取的标签数量
        :return: 关键词列表

        使用
        ------
        - 给定要提前关键词的数量及文件即可得到该文本的关键词

        注意
        ------
        - 请先使用 clean_text方法来获取清洗好的词汇表
        - 关键词列表按权重逆序排列, 权重越大越关键
        - 若最终文本的关键词数量达不到要求提取的数量, 则返回所有已经提取到的关键词
        """

        words_frequency = self.get_words_frequency(cleaned_wordlist)
        words_weight = self.get_words_weight(cleaned_wordlist)

        words_filter = [word for word in words_weight if not word.isnumeric() and 'http' not in word]
        tags = [word for word in words_frequency if word in words_filter]

        return tags[:min(tag_number, len(tags))]

    @staticmethod
    def get_words_frequency(cleaned_wordlist: list[str]) -> dict[str, int]:
        """
        获得词汇的词频排序

        :param cleaned_wordlist: 处理后的词列表
        :return: 倒序排列的词频字典

        使用
        ------
        - 将清洗好的词汇表作为本方法的分析参数, 最终获得按词频倒序排列的字典

        注意
        ------
        - 请先使用 clean_text方法来获取清洗好的词汇表
        """

        words_frequency = {}
        for word in cleaned_wordlist:
            if len(word) != 1:
                words_frequency[word] = words_frequency.get(word, 0) + 1

        return dict(sorted(words_frequency.items(), key = lambda x: x[1], reverse = True))

    @staticmethod
    def get_words_weight(cleaned_wordlist: list[str]) -> dict[str, float]:
        """
        获得词汇的TF-IDF权重排序

        :param cleaned_wordlist: 处理后的词列表
        :return: 倒序排列的词汇TF-IDF权重字典

        使用
        ------
        - 将清洗好的词汇表作为本方法的分析参数, 最终获得按TF-IDF权重倒序排列的字典

        注意
        ------
        - 请先使用 clean_text方法来获取清洗好的词汇表
        """

        # 获得词表和语料库
        corpus = [' '.join(cleaned_wordlist)]

        # 计算TF-IDF向量
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(corpus)

        # 获得权重字典
        words_weight = dict(zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0]))
        return dict(sorted(words_weight.items(), key = lambda x: x[1], reverse = True))
