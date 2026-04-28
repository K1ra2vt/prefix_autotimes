#!/usr/bin/env python
"""
测试AMP模式下的Cross-Attention实现
"""
import torch
from models.prefix import CrossAttentionPrefixEnhancer

def test_amp_mode():
    """测试在AMP模式下cross-attention是否正常工作"""
    print("测试AMP模式下的Cross-Attention...")

    # 创建enhancer
    enhancer = CrossAttentionPrefixEnhancer(
        hidden_dim=768,
        num_heads=8,
        dropout=0.1
    )

    # 创建测试数据
    batch_size = 4
    prefix_len = 4
    seq_len = 24

    prefix_embeds = torch.randn(batch_size, prefix_len, 768)
    timeseries_embeds = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len)

    # 测试float32模式
    print("\n1. 测试float32模式...")
    enhancer.float()
    output_fp32 = enhancer(prefix_embeds, timeseries_embeds, attention_mask)
    print(f"✓ FP32输出形状: {output_fp32.shape}, dtype: {output_fp32.dtype}")

    # 测试float16模式（手动）
    print("\n2. 测试float16模式...")
    enhancer.half()
    prefix_embeds_fp16 = prefix_embeds.half()
    timeseries_embeds_fp16 = timeseries_embeds.half()

    try:
        output_fp16 = enhancer(prefix_embeds_fp16, timeseries_embeds_fp16, attention_mask)
        print(f"✓ FP16输出形状: {output_fp16.shape}, dtype: {output_fp16.dtype}")
    except RuntimeError as e:
        print(f"✗ FP16模式失败: {e}")
        return False

    # 测试AMP自动混合精度
    print("\n3. 测试AMP自动混合精度...")
    enhancer.float()  # 回到float32
    prefix_embeds = prefix_embeds.float()
    timeseries_embeds = timeseries_embeds.float()

    scaler = torch.cuda.amp.GradScaler()

    # 模拟forward pass with autocast
    try:
        with torch.cuda.amp.autocast():
            output_amp = enhancer(prefix_embeds, timeseries_embeds, attention_mask)
            print(f"✓ AMP输出形状: {output_amp.shape}, dtype: {output_amp.dtype}")

        # 测试backward
        loss = output_amp.sum()
        scaler.scale(loss).backward()
        print(f"✓ AMP backward成功")

    except RuntimeError as e:
        print(f"✗ AMP模式失败: {e}")
        return False

    print("\n✓ 所有AMP测试通过！")
    return True

if __name__ == "__main__":
    if torch.cuda.is_available():
        success = test_amp_mode()
        exit(0 if success else 1)
    else:
        print("CUDA不可用，跳过AMP测试")
        exit(0)
