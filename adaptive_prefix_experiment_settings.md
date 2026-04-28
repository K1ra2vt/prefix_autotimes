# 自适应Prefix门控方法实验设置

## 1. 实验环境

### 1.1 硬件环境
- **GPU配置**: 单张 NVIDIA GPU（根据实验设置可配置）
- **精度模式**: 混合精度训练（Automatic Mixed Precision, AMP）
- **数据加载并行度**: 10 workers

### 1.2 软件环境
- **深度学习框架**: PyTorch
- **预训练模型**: Transformers库 (GPT-2)
- **优化器**: Adam

## 2. 模型架构

### 2.1 基础模型
- **骨干网络**: GPT-2 (AutoTimes_Gpt2)
- **冻结策略**: 冻结LLM除Prefix外的所有参数 (`freeze_llm_except_prefix=true`)
- **注入模式**: 深层注入 (`prefix_injection_mode=deep`)

### 2.2 Prefix设计
自适应Prefix采用**可学习的静态层门控**与**动态强度调整**相结合的机制：

#### 静态层门控机制
- **功能**: 为每个Transformer层学习独立的重要性权重
- **初始化策略**: 采用高斯分布初始化，使中间层具有更高的初始权重
  \[
  \text{init\_gates}[i] = 0.5 + 0.3 \times \exp\left(-\frac{(i - \text{mid})^2}{(\text{num\_layers}/3)^2}\right)
  \]
  其中 \(\text{mid} = \text{num\_layers} // 2\)

#### 动态强度调整
- **强度预测器**: 两层MLP网络
  - 第一层: \(512 \rightarrow 256\) 维
  - 第二层: \(256 \rightarrow \text{num\_layers}\) 维
  - 激活函数: GELU
  - 包含LayerNorm和Dropout(0.1)
- **功能**: 根据输入时序特征动态调整每一层的Prefix注入强度

### 2.3 Prefix生成网络结构
- **Prefix长度**: 4 tokens
- **Token嵌入**: 词嵌入维度 512
- **位置嵌入**: 最大长度 64
- **Token处理MLP**: 2层网络
  - 隐藏层维度: 512
  - 激活函数: GELU
  - 包含LayerNorm和Dropout(0.1)
- **Prefix生成器**: 分离生成Key和Value
  - 输出维度: \(\text{num\_layers} \times \text{prefix\_length} \times \text{hidden\_dim}\)

### 2.4 门控参数配置
- **门控最小值**: 0.1（避免完全关闭）
- **温度参数**: 0.1（控制门控的尖锐程度）
- **特征聚合**: 自适应平均池化

## 3. 训练配置

### 3.1 差分学习率策略
采用**分组学习率**策略，对不同模块设置不同的学习率：

| 模块 | 学习率 | 说明 |
|------|--------|------|
| Prefix参数 | 1e-4 | Prefix生成网络及门控机制 |
| Encoder-Decoder | 5e-4 | 编码器和解码器部分 |
| 其他参数 | 1e-4 | 基础学习率 |

### 3.2 训练超参数
- **Batch Size**:
  - Weather数据集: 2048
  - ECL数据集: 1536
- **训练轮数**: 10 epochs
- **权重衰减**: 1e-5
- **学习率调度**: Type2调度器
- **梯度裁剪**: 未明确指定（使用默认值）

### 3.3 序列配置
| 参数 | 值 | 说明 |
|------|-----|------|
| seq_len | 672 | 输入序列长度 |
| label_len | 576 | 标签序列长度 |
| token_len | 96 | Token长度 |
| max_prompt_length | 64 | 最大提示长度 |

### 3.4 测试配置
- **测试预测长度**: [96, 192, 336, 720]
- **测试方式**: 单次前向传播（无梯度）

## 4. 数据集配置

### 4.1 使用的数据集
1. **Weather**: 气象测量数据
2. **ECL (Electricity)**: 电力负载数据

### 4.2 数据预处理
- **标准化**: StandardScaler
- **划分比例**:
  - 训练集: 70%
  - 测试集: 20%
  - 验证集: 10%

## 5. 模型特点与创新

### 5.1 自适应选择性注入
1. **层级门控**: 不同层可以根据学习到的权重选择性地注入Prefix
2. **动态调整**: 注入强度可以根据输入特征实时调整
3. **稀疏性优势**: 某些层门控接近0时几乎不注入，节省计算资源

### 5.2 训练策略
- **参数高效**: 仅训练Prefix相关参数，冻结LLM主干
- **差分学习率**: 针对不同模块采用不同学习率
- **混合精度**: 使用AMP加速训练并节省显存

## 6. 实验输出

模型训练完成后，会在以下预测长度上进行评估：
- 96步预测（短期）
- 192步预测（中期）
- 336步预测（中长期）
- 720步预测（长期）

每个预测长度独立测试，评估指标包括MSE和MAE。

## 7. 关键超参数总结

| 超参数 | 值 | 超参数 | 值 |
|--------|-----|--------|-----|
| prefix_length | 4 | prefix_mlp_hidden | 512 |
| prefix_mlp_layers | 2 | adaptive_min_gate_value | 0.1 |
| adaptive_gate_temperature | 0.1 | learning_rate | 1e-4 |
| prefix_learning_rate | 1e-4 | encoder_decoder_learning_rate | 5e-4 |
| train_epochs | 10 | batch_size | 1536-2048 |
| weight_decay | 1e-5 | num_workers | 10 |
