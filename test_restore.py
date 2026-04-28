#!/usr/bin/env python
"""验证恢复到原始版本"""
import sys
import torch

print("测试恢复到原始Prefix版本...")

# 测试1: 导入
try:
    from models.AutoTimes_Gpt2 import Model
    from models.prefix import TokenBasedConditionalPrefix
    print("✓ 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: 检查不再有cross_attn相关属性
try:
    class Config:
        def __init__(self):
            self.token_len = 96
            self.use_multi_gpu = False
            self.local_rank = 0
            self.gpu = 0
            self.llm_ckp_dir = '/home/u4_3090_4/baseModel_gpt2'
            self.mix_embeds = False
            self.use_prefix = True
            self.prefix_injection_mode = 'deep'
            self.prefix_length = 4
            self.prefix_mlp_hidden = 512
            self.prefix_mlp_layers = 2
            self.max_prompt_length = 64
            self.dropout = 0.1
            self.prefix_init_with_zeros = False
            self.mlp_hidden_layers = 2
            self.mlp_hidden_dim = 256
            self.mlp_activation = 'tanh'

    configs = Config()
    model = Model(configs)

    # 检查属性
    assert hasattr(model, 'use_prefix'), "缺少use_prefix属性"
    assert hasattr(model, 'prefix_injection_mode'), "缺少prefix_injection_mode属性"
    assert not hasattr(model, 'use_cross_attention_prefix'), "不应该有use_cross_attention_prefix属性"

    # 检查prefix类型
    from models.prefix import TokenBasedConditionalPrefix
    assert isinstance(model.prefix, TokenBasedConditionalPrefix), "prefix类型应该是TokenBasedConditionalPrefix"
    assert type(model.prefix).__name__ == 'TokenBasedConditionalPrefix', f"prefix类型错误: {type(model.prefix).__name__}"

    print("✓ 模型初始化正确")
    print(f"  - prefix类型: {type(model.prefix).__name__}")
    print(f"  - 不包含cross_attn相关代码")

    print("\n✓ 所有测试通过！已恢复到原始版本。")
    sys.exit(0)

except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
