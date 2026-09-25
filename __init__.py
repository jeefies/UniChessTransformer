"""T 的模型预设表（``Transformer.model`` 里定义）与配置构造。

集中放在这里是为了让配置与测试都能 ``from Transformer import PRESETS``，避免各处
重复 import 模型模块（模型模块 import torch，测试里只想拿预设名时不必吃下这份开销）。
"""
from .model import PRESETS, StratifiedChessTransformer, TransformerConfig, stratified_20m

__all__ = ["PRESETS", "StratifiedChessTransformer", "TransformerConfig", "stratified_20m"]
