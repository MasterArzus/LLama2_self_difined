from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768, # 模型维度
            n_layers: int = 12, # Transformer的层数
            n_heads: int = 16, # 注意力机制的头数
            n_kv_heads: int = 8, # 键值头的数量
            vocab_size: int = 6144, # 词汇表大小
            hidden_dim: int = None, # 隐藏层维度
            multiple_of: int = 64, 
            norm_eps: float = 1e-5, # 归一化层的eps
            max_seq_len: int = 512, # 最大序列长度
            dropout: float = 0.0, # dropout概率
            flash_attn: bool = True, # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        # 这个地方是把PretrainedConfig这个父类的函数加载一下
        super().__init__(**kwargs)


        '''
        **kwargs是为了兼容其他可能的参数，这样可以在初始化时传入额外的配置参数
        example:
        def func(a, **kwargs):
            print(a)
            print(kwargs)
        func(1, b=2, c=3)  # 输出：1 {'b': 2, 'c': 3}

        传递进的额外参数通过键值对表示，函数内可以通过键值对提取
        example:
        def func(**kwargs):
            # 直接访问某个键
            value1 = kwargs['key']  # 如果'key'不存在会报错
            # 推荐用get方法，避免报错
            value2 = kwargs.get('key', '默认值')  # 如果'key'不存在则返回'默认值'
            print(value1)
            print(value2)


        对比*args和**kwargs的区别：
        - *args用于接收位置参数，**kwargs用于接收关键字参数。
        - *args会将多余的位置参数打包成一个元组，而**kwargs会将多余的关键字参数打包成一个字典。
        examples:
        def func(*args):
            # 获取第一个参数
            first = args[0]
            # 获取全部参数
            print(args)
            # 遍历所有参数
            for value in args:
                print(value)

        '''
