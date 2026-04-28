import torch
import torch.nn as nn
import math
from transformers import GPT2Tokenizer
import json
import time

# debug log path for runtime instrumentation (do not remove)
LOG_PATH = "/home/u4_3090_4/.cursor/debug.log"

def create_structured_prompt_template(dataset_name, seq_len, pred_len, stats, domain_info="",
                                     top_lags=None):
    def fmt(x):
        try:
            return f"{x:.3f}"
        except Exception:
            return str(x) if x is not None else "N/A"

    min_val = fmt(stats.get('min', stats.get('min_val', None)))
    max_val = fmt(stats.get('max', stats.get('max_val', None)))
    median_val = fmt(stats.get('median', stats.get('median_val', stats.get('mean', None))))
    trend = stats.get('trend', None)
    if isinstance(trend, (float, int)):
        trend_str = "upward" if trend > 0 else ("downward" if trend < 0 else "stable")
    else:
        trend_str = str(trend) if trend is not None else "N/A"

    top_lags_str = "N/A"
    if top_lags:
        try:
            top_lags_str = ", ".join([str(x) for x in top_lags[:5]])
        except Exception:
            top_lags_str = str(top_lags)

    default_domain = {
        'ETTh1': "Electricity Transformer Temperature (monitoring transformer temperatures and power loads)",
        'ETTm1': "Electricity Transformer Temperature (different frequency)",
        'Traffic': "Urban traffic flow monitoring",
        'Weather': "Meteorological measurements"
    }.get(dataset_name, domain_info or "Time series data")

    prompt = f"""The {dataset_name} dataset: {default_domain}
[Domain]: {domain_info or default_domain}
[Instruction]: Predict the next {pred_len} steps given the previous {seq_len} steps information attached
[Statistics]: The input has a minimum of {min_val}, a maximum of {max_val}, and a median of {median_val}. The overall trend is {trend_str}. The top five lags are {top_lags_str}.
"""
    return prompt


def tokenize_structured_prompt(prompt_text, tokenizer, max_length=64):
    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", 64)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)


def extract_structured_context_features(time_series_data, dataset_name='ETT', seq_len=672, pred_len=96, tokenizer=None):
    if len(time_series_data.shape) == 2:
        batch_size, seq_len_actual = time_series_data.shape
        n_vars = 1
        time_series_data = time_series_data.unsqueeze(-1)
    else:
        batch_size, seq_len_actual, n_vars = time_series_data.shape

    numeric_features = []

    for batch_idx in range(batch_size):
        sample_data = time_series_data[batch_idx]
        data_mean = sample_data.mean(dim=0)
        data_std = sample_data.std(dim=0)
        data_min = sample_data.min(dim=0)[0]
        data_max = sample_data.max(dim=0)[0]
        trend = sample_data[-1] - sample_data[0]
        sample_features = torch.cat([data_mean, data_std, data_min, data_max, trend])
        numeric_features.append(sample_features)

    numeric_features = torch.stack(numeric_features)

    text_prompts = []
    domain_specific_info = ""

    for batch_idx in range(batch_size):
        sample_data = time_series_data[batch_idx]
        stats = {
            'mean': sample_data.mean().item(),
            'std': sample_data.std().item(),
            'min': sample_data.min().item(),
            'max': sample_data.max().item(),
            'trend': (sample_data[-1] - sample_data[0]).mean().item()
        }
        if dataset_name.upper().startswith('ETT'):
            if n_vars >= 7:
                hufl_mean = sample_data[:, 0].mean().item()
                hull_mean = sample_data[:, 1].mean().item()
                ot_trend = (sample_data[-1, -1] - sample_data[0, -1]).item()
                domain_specific_info = f"""Variables: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
Load Ratio: {hufl_mean/(hull_mean+1e-6):.3f}
Oil Temp Trend: {ot_trend:.3f}
Frequency: hourly"""
        elif dataset_name.upper().startswith('TRAFFIC'):
            domain_specific_info = f"""Variables: {n_vars} traffic sensors
Frequency: hourly
Domain: Urban traffic flow"""
        elif dataset_name.upper().startswith('WEATHER'):
            domain_specific_info = f"""Variables: meteorological measurements
Frequency: 10 minutes
Domain: Weather forecasting"""
        elif dataset_name.upper().startswith('ELECTRICITY'):
            domain_specific_info = f"""Variables: {n_vars} electricity consumption channels
Frequency: hourly
Domain: Power consumption"""

        prompt = create_structured_prompt_template(
            dataset_name=dataset_name,
            seq_len=seq_len,
            pred_len=pred_len,
            stats=stats,
            domain_info=domain_specific_info
        )
        text_prompts.append(prompt)

    if tokenizer is not None:
        tokenized_prompts = []
        attention_masks = []
        for prompt in text_prompts:
            tokens, mask = tokenize_structured_prompt(prompt, tokenizer)
            tokenized_prompts.append(tokens)
            attention_masks.append(mask)
        tokenized_prompts = torch.stack(tokenized_prompts)
        attention_masks = torch.stack(attention_masks)
        return {
            'tokens': tokenized_prompts,
            'attention_masks': attention_masks,
            'text_prompts': text_prompts,
            'numeric_features': numeric_features,
            'max_length': tokenized_prompts.shape[-1]
        }
    else:
        return {
            'numeric_features': numeric_features,
            'text_prompts': text_prompts,
            'feature_dim': numeric_features.shape[-1] if len(numeric_features.shape) > 1 else 1
        }


class ConditionalPrefix(nn.Module):
    def __init__(self,
                 context_dim,
                 hidden_dim,
                 prefix_length,
                 num_layers,
                 mlp_hidden_dim=512,
                 mlp_layers=2,
                 dropout=0.1,
                 activation='gelu'):
        super(ConditionalPrefix, self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation {activation} not supported")
        layers = [nn.Linear(self.context_dim, self.mlp_hidden_dim),
                  self.activation,
                  nn.LayerNorm(self.mlp_hidden_dim),
                  nn.Dropout(self.dropout)]
        for i in range(self.mlp_layers - 1):
            layers += [nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
                       self.activation,
                       nn.LayerNorm(self.mlp_hidden_dim),
                       nn.Dropout(self.dropout)]
        final_out_dim = self.num_layers * self.prefix_length * self.hidden_dim
        layers += [nn.Linear(self.mlp_hidden_dim, final_out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, context_features):
        batch_size = context_features.shape[0]
        prefix_flat = self.mlp(context_features)
        prefix = prefix_flat.view(batch_size, self.num_layers, self.prefix_length, self.hidden_dim)
        return prefix


class TokenBasedConditionalPrefix(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_length,
                 hidden_dim,
                 prefix_length,
                 num_layers,
                 mlp_hidden_dim=512,
                 mlp_layers=2,
                 dropout=0.1,
                 activation='gelu',
                 init_with_zeros=False):
        super(TokenBasedConditionalPrefix, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.init_with_zeros = init_with_zeros
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation {activation} not supported")
        self.token_embedding = nn.Embedding(vocab_size, mlp_hidden_dim)
        self.position_embedding = nn.Embedding(max_length, mlp_hidden_dim)
        token_layers = []
        for i in range(mlp_layers):
            token_layers += [nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                             self.activation,
                             nn.LayerNorm(mlp_hidden_dim),
                             nn.Dropout(dropout)]
        self.token_processor = nn.Sequential(*token_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        final_out_dim = self.num_layers * self.prefix_length * self.hidden_dim
        self.prefix_generator_k = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            self.activation,
            nn.LayerNorm(mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, final_out_dim)
        )
        self.prefix_generator_v = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            self.activation,
            nn.LayerNorm(mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, final_out_dim)
        )
            # Apply zero initialization if requested
        if self.init_with_zeros:
            self.apply(self._init_weights_zero)

    def _init_weights_zero(self, module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.zeros_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens, attention_mask=None, patch_pooled=None, patch_stats=None):
        batch_size = tokens.shape[0]
        token_embeds = self.token_embedding(tokens)
        positions = torch.arange(self.max_length, device=tokens.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        pos_embeds = pos_embeds.expand(batch_size, -1, -1)
        combined_embeds = token_embeds + pos_embeds # 不懂为什么这个要加上pos
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            combined_embeds = combined_embeds * mask_expanded
        # region agent log
        try:
            with open(LOG_PATH, "a") as _f:
                _f.write(json.dumps({
                    "sessionId":"debug-session","runId":"run1","hypothesisId":"H1",
                    "location":"prefix.py:forward:after_combined_embeds",
                    "message":"combined_embeds_shapes",
                    "data":{
                        "token_embeds_shape": list(token_embeds.shape),
                        "pos_embeds_shape": list(pos_embeds.shape),
                        "combined_embeds_shape": list(combined_embeds.shape),
                        "attention_mask_shape": None if attention_mask is None else list(attention_mask.shape)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\\n")
        except Exception:
            pass
        # endregion agent log
        processed_tokens = self.token_processor(combined_embeds)
        # region agent log
        try:
            with open(LOG_PATH, "a") as _f:
                _f.write(json.dumps({
                    "sessionId":"debug-session","runId":"run1","hypothesisId":"H2",
                    "location":"prefix.py:forward:after_processed_tokens",
                    "message":"processed_tokens_info",
                    "data":{
                        "processed_tokens_shape": list(processed_tokens.shape),
                        "processed_tokens_dim": processed_tokens.dim(),
                        "processed_tokens_dtype": str(processed_tokens.dtype)
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\\n")
        except Exception:
            pass
        # endregion agent log
        pooled = self.pooling(processed_tokens.permute(0, 2, 1)).squeeze(-1)
        prefix_flat_k = self.prefix_generator_k(pooled)
        prefix_flat_v = self.prefix_generator_v(pooled)
        prefix_k = prefix_flat_k.view(batch_size, self.num_layers, self.prefix_length, self.hidden_dim)
        prefix_v = prefix_flat_v.view(batch_size, self.num_layers, self.prefix_length, self.hidden_dim)

        prefix = torch.stack([prefix_k, prefix_v], dim=2)  # [B, L, 2, p_len, H]
        return prefix


class CrossAttentionPrefixEnhancer(nn.Module):
    """
    Cross-Attention模块，让prefix和时序embedding进行交互增强
    类似于FLAN-T5的设计，让context和query相互attend
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionPrefixEnhancer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"

        self.cross_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn_v = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 门控机制，控制融合程度
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, prefix_embeds, timeseries_embeds, attention_mask=None):
        """
        Args:
            prefix_embeds: [batch, prefix_len, hidden_dim]
            timeseries_embeds: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] 可选的时序数据mask
        Returns:
            enhanced_prefix: [batch, prefix_len, hidden_dim]
        """
        batch_size = prefix_embeds.size(0)

        # 计算cross-attention: prefix作为query，时序数据作为key和value
        Q = self.cross_attn_q(prefix_embeds)  # [B, p_len, H]
        K = self.cross_attn_k(timeseries_embeds)  # [B, seq_len, H]
        V = self.cross_attn_v(timeseries_embeds)  # [B, seq_len, H]

        # reshape为多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, p_len, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_len, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq_len, head_dim]

        # 计算attention分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, p_len, seq_len]

        # 应用mask（如果有）
        if attention_mask is not None:
            # attention_mask: [B, seq_len] -> [B, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 根据数据类型选择合适的mask值，避免float16溢出
            if attn_scores.dtype == torch.float16:
                mask_value = -1e4  # float16安全范围内
            else:
                mask_value = -1e9  # float32可以安全使用
            attn_scores = attn_scores.masked_fill(mask == 0, mask_value)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用attention到value
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, p_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, p_len, heads, head_dim]
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)  # [B, p_len, H]

        # 输出投影
        enhanced = self.out_proj(attn_output)  # [B, p_len, H]

        # 残差连接和LayerNorm
        enhanced = self.layer_norm(enhanced + prefix_embeds)

        # 门控融合：原始prefix和增强prefix的加权组合
        gate_value = self.gate(torch.cat([prefix_embeds, enhanced], dim=-1))  # [B, p_len, H]
        final_output = gate_value * enhanced + (1 - gate_value) * prefix_embeds

        return final_output


class CrossAttentionTokenBasedConditionalPrefix(nn.Module):
    """
    增强版TokenBasedConditionalPrefix，整合Cross-Attention机制
    让生成的prefix能够与时序数据进行交互，增强时序感知能力
    """
    def __init__(self,
                 vocab_size,
                 max_length,
                 hidden_dim,
                 prefix_length,
                 num_layers,
                 mlp_hidden_dim=512,
                 mlp_layers=2,
                 dropout=0.1,
                 activation='gelu',
                 init_with_zeros=False,
                 use_cross_attention=True,
                 cross_attn_heads=8):
        super(CrossAttentionTokenBasedConditionalPrefix, self).__init__()
        self.use_cross_attention = use_cross_attention

        # 基础的prefix生成器
        self.base_prefix = TokenBasedConditionalPrefix(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=hidden_dim,
            prefix_length=prefix_length,
            num_layers=num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_layers=mlp_layers,
            dropout=dropout,
            activation=activation,
            init_with_zeros=init_with_zeros
        )

        # Cross-Attention增强器（如果启用）
        if use_cross_attention:
            self.cross_enhancer = CrossAttentionPrefixEnhancer(
                hidden_dim=hidden_dim,
                num_heads=cross_attn_heads,
                dropout=dropout
            )
        else:
            self.cross_enhancer = None

        # 用于将增强后的prefix转换回K/V格式
        final_out_dim = num_layers * prefix_length * hidden_dim
        self.enhanced_to_k = nn.Linear(hidden_dim, hidden_dim)
        self.enhanced_to_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens, attention_mask=None, timeseries_embeds=None, timeseries_attn_mask=None):
        """
        Args:
            tokens: [batch, max_length] tokenized prompt
            attention_mask: [batch, max_length] prompt attention mask
            timeseries_embeds: [batch, seq_len, hidden_dim] 时序数据embedding（用于cross-attention）
            timeseries_attn_mask: [batch, seq_len] 时序数据attention mask
        Returns:
            prefix: [batch, num_layers, 2, prefix_length, hidden_dim]
        """
        batch_size = tokens.shape[0]

        # 生成基础prefix
        base_prefix = self.base_prefix(tokens, attention_mask)

        if self.use_cross_attention and timeseries_embeds is not None:
            # 对每一层的prefix都应用cross-attention增强
            num_layers = base_prefix.size(1)
            prefix_length = base_prefix.size(2)
            hidden_dim = base_prefix.size(4)

            enhanced_prefix_list = []

            for layer_idx in range(num_layers):
                # 提取当前层的K和V
                layer_k = base_prefix[:, layer_idx, 0, :, :]  # [B, p_len, H]
                layer_v = base_prefix[:, layer_idx, 1, :, :]  # [B, p_len, H]

                # 对K应用cross-attention增强（让prefix关注时序数据）
                enhanced_k = self.cross_enhancer(
                    layer_k, timeseries_embeds, timeseries_attn_mask
                )
                # 对V也应用cross-attention增强
                enhanced_v = self.cross_enhancer(
                    layer_v, timeseries_embeds, timeseries_attn_mask
                )

                # 投影到正确的维度
                enhanced_k = self.enhanced_to_k(enhanced_k)
                enhanced_v = self.enhanced_to_v(enhanced_v)

                # 组合K和V
                layer_kv = torch.stack([enhanced_k, enhanced_v], dim=1)  # [B, 2, p_len, H]
                enhanced_prefix_list.append(layer_kv)

            # 堆叠所有层
            enhanced_prefix = torch.stack(enhanced_prefix_list, dim=1)  # [B, L, 2, p_len, H]
            return enhanced_prefix
        else:
            return base_prefix


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation)线性层
    在原始线性层旁边添加低秩分解路径
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # 原始线性层（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        # LoRA低秩分解
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = self.alpha / self.rank

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 原始路径
        result = self.linear(x)

        # LoRA路径
        lora_out = self.dropout(x)
        lora_out = lora_out @ self.lora_A  # [B, rank]
        lora_out = lora_out @ self.lora_B  # [B, out_features]
        lora_out = lora_out * self.scaling

        return result + lora_out

    def train(self, mode: bool = True):
        # 训练时冻结原始linear，只训练LoRA参数
        self.linear.weight.requires_grad = False
        self.lora_A.requires_grad = mode
        self.lora_B.requires_grad = mode
        return super().train(mode)


class LoRALayerwiseTokenBasedConditionalPrefix(nn.Module):
    """
    结合LoRA和Layer-wise的Prefix生成器

    特点：
    1. Layer-wise: 将层分为若干组，每组有独立的prefix生成器
    2. LoRA: 在关键线性层使用LoRA降低参数量和训练难度
    3. 灵活的prefix长度配置
    """
    def __init__(self,
                 vocab_size,
                 max_length,
                 hidden_dim,
                 num_layers,
                 prefix_length=4,
                 mlp_hidden_dim=512,
                 mlp_layers=2,
                 dropout=0.1,
                 activation='gelu',
                 init_with_zeros=False,
                 # LoRA参数
                 use_lora=True,
                 lora_rank=8,
                 lora_alpha=16,
                 # Layer-wise参数
                 num_layer_groups=3,  # 将层分为几组
                 layerwise_prefix_lengths=None,  # 每组的prefix长度，如[2, 4, 6]
                 ):
        super(LoRALayerwiseTokenBasedConditionalPrefix, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.init_with_zeros = init_with_zeros
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation {activation} not supported")

        # Token和Position编码
        self.token_embedding = nn.Embedding(vocab_size, mlp_hidden_dim)
        self.position_embedding = nn.Embedding(max_length, mlp_hidden_dim)

        # Token处理MLP（共享）
        token_processor_layers = []
        for i in range(mlp_layers):
            token_processor_layers += [nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                                      self.activation,
                                      nn.LayerNorm(mlp_hidden_dim),
                                      nn.Dropout(dropout)]
        self.token_processor = nn.Sequential(*token_processor_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Layer-wise配置
        self.num_layer_groups = num_layer_groups
        if layerwise_prefix_lengths is None:
            # 默认：浅层短，深层长
            if num_layer_groups == 2:
                layerwise_prefix_lengths = [2, 4]  # 前半层2个token，后半层4个
            elif num_layer_groups == 3:
                layerwise_prefix_lengths = [2, 4, 6]  # 渐进式增长
            elif num_layer_groups == 4:
                layerwise_prefix_lengths = [2, 3, 4, 6]
            else:
                # 平分
                base_len = prefix_length
                layerwise_prefix_lengths = [base_len] * num_layer_groups

        assert len(layerwise_prefix_lengths) == num_layer_groups, \
            f"layerwise_prefix_lengths长度({len(layerwise_prefix_lengths)})必须等于num_layer_groups({num_layer_groups})"

        self.layerwise_prefix_lengths = layerwise_prefix_lengths

        # 计算每组的层索引
        layers_per_group = num_layers // num_layer_groups
        self.layer_groups = []
        for i in range(num_layer_groups):
            start_layer = i * layers_per_group
            end_layer = (i + 1) * layers_per_group if i < num_layer_groups - 1 else num_layers
            self.layer_groups.append((start_layer, end_layer))

        # 为每组创建独立的prefix生成器
        self.group_prefix_generators = nn.ModuleList()
        for group_idx in range(num_layer_groups):
            group_prefix_len = layerwise_prefix_lengths[group_idx]
            layers_in_group = self.layer_groups[group_idx][1] - self.layer_groups[group_idx][0]
            group_out_dim = layers_in_group * group_prefix_len * hidden_dim

            if use_lora:
                # 使用LoRA版本的生成器
                generator_k = nn.Sequential(
                    LoRALinear(mlp_hidden_dim, mlp_hidden_dim, lora_rank, lora_alpha, dropout),
                    self.activation,
                    nn.LayerNorm(mlp_hidden_dim),
                    nn.Dropout(dropout),
                    LoRALinear(mlp_hidden_dim, group_out_dim, lora_rank, lora_alpha, dropout)
                )
                generator_v = nn.Sequential(
                    LoRALinear(mlp_hidden_dim, mlp_hidden_dim, lora_rank, lora_alpha, dropout),
                    self.activation,
                    nn.LayerNorm(mlp_hidden_dim),
                    nn.Dropout(dropout),
                    LoRALinear(mlp_hidden_dim, group_out_dim, lora_rank, lora_alpha, dropout)
                )
            else:
                # 使用标准版本的生成器
                generator_k = nn.Sequential(
                    nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                    self.activation,
                    nn.LayerNorm(mlp_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_dim, group_out_dim)
                )
                generator_v = nn.Sequential(
                    nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                    self.activation,
                    nn.LayerNorm(mlp_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_dim, group_out_dim)
                )

            self.group_prefix_generators.append(nn.ModuleDict({
                'generator_k': generator_k,
                'generator_v': generator_v
            }))

        # Zero初始化（如果需要）
        if init_with_zeros:
            self.apply(self._init_weights_zero)

    def _init_weights_zero(self, module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LoRALinear):
            nn.init.zeros_(module.linear.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.zeros_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens, attention_mask=None, patch_pooled=None):
        batch_size = tokens.shape[0]

        # 处理tokens（共享）
        token_embeds = self.token_embedding(tokens)
        positions = torch.arange(self.max_length, device=tokens.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        pos_embeds = pos_embeds.expand(batch_size, -1, -1)
        combined_embeds = token_embeds + pos_embeds

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            combined_embeds = combined_embeds * mask_expanded

        processed_tokens = self.token_processor(combined_embeds)
        pooled = self.pooling(processed_tokens.permute(0, 2, 1)).squeeze(-1)

        # 为每个层组生成prefix
        max_prefix_len = max(self.layerwise_prefix_lengths)

        # 创建输出的tensor（直接填充到统一长度）
        prefix_output_k = []
        prefix_output_v = []

        for group_idx in range(self.num_layer_groups):
            start_layer, end_layer = self.layer_groups[group_idx]
            group_prefix_len = self.layerwise_prefix_lengths[group_idx]

            # 生成该组的K和V
            group_generator = self.group_prefix_generators[group_idx]
            group_prefix_k_flat = group_generator['generator_k'](pooled)
            group_prefix_v_flat = group_generator['generator_v'](pooled)

            # Reshape
            layers_in_group = end_layer - start_layer
            group_prefix_k = group_prefix_k_flat.view(batch_size, layers_in_group, group_prefix_len, self.hidden_dim)
            group_prefix_v = group_prefix_v_flat.view(batch_size, layers_in_group, group_prefix_len, self.hidden_dim)

            # 为该组的每一层添加prefix（如果需要则填充）
            for layer_idx in range(start_layer, end_layer):
                relative_idx = layer_idx - start_layer
                layer_k = group_prefix_k[:, relative_idx, :, :]  # [B, group_p_len, H]
                layer_v = group_prefix_v[:, relative_idx, :, :]  # [B, group_p_len, H]

                # 填充到max_prefix_len
                if group_prefix_len < max_prefix_len:
                    pad_size = max_prefix_len - group_prefix_len
                    padding_k = torch.zeros(batch_size, pad_size, self.hidden_dim,
                                          device=layer_k.device, dtype=layer_k.dtype)
                    padding_v = torch.zeros(batch_size, pad_size, self.hidden_dim,
                                          device=layer_v.device, dtype=layer_v.dtype)
                    layer_k = torch.cat([layer_k, padding_k], dim=1)  # [B, max_p_len, H]
                    layer_v = torch.cat([layer_v, padding_v], dim=1)

                prefix_output_k.append(layer_k)
                prefix_output_v.append(layer_v)

        # 堆叠所有层 [L, B, max_p_len, H]
        all_k = torch.stack(prefix_output_k, dim=0)
        all_v = torch.stack(prefix_output_v, dim=0)

        # 转换为 [B, L, 2, max_p_len, H] 格式
        final_prefix = torch.stack([all_k, all_v], dim=2)  # [L, B, 2, max_p_len, H]
        final_prefix = final_prefix.permute(1, 0, 2, 3, 4)  # [B, L, 2, max_p_len, H]

        return final_prefix


class AdaptiveTokenBasedPrefix(nn.Module):
    """
    自适应选择性注入的Prefix生成器
    
    特点：
    1. 可学习的静态层门控：学习哪些层应该注入prefix
    2. 动态强度调整：根据输入时序特征动态调整每层注入强度
    3. 稀疏性：某些层门控接近0时几乎不注入，节省计算
    
    输出格式：[B, L, 2, prefix_len, H]，但部分层可能是0（不注入）
    """
    def __init__(self,
                 vocab_size,
                 max_length,
                 hidden_dim,
                 prefix_length,
                 num_layers,
                 mlp_hidden_dim=512,
                 mlp_layers=2,
                 dropout=0.1,
                 activation='gelu',
                 init_with_zeros=False,
                 # 自适应参数
                 use_static_gating=True,
                 use_dynamic_intensity=True,
                 min_gate_value=0.1,  # 门控最小值，避免完全关闭
                 temperature=0.1):  # 温度参数，控制门控的尖锐程度
        super(AdaptiveTokenBasedPrefix, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.init_with_zeros = init_with_zeros
        self.use_static_gating = use_static_gating
        self.use_dynamic_intensity = use_dynamic_intensity
        self.min_gate_value = min_gate_value
        self.temperature = temperature
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation {activation} not supported")
        
        # Token和Position编码（与原版相同）
        self.token_embedding = nn.Embedding(vocab_size, mlp_hidden_dim)
        self.position_embedding = nn.Embedding(max_length, mlp_hidden_dim)
        
        # Token处理MLP（与原版相同）
        token_layers = []
        for i in range(mlp_layers):
            token_layers += [nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                           self.activation,
                           nn.LayerNorm(mlp_hidden_dim),
                           nn.Dropout(dropout)]
        self.token_processor = nn.Sequential(*token_layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # ===== 自适应门控机制 =====
        if use_static_gating:
            # 可学习的静态层门控 - 每个层一个参数，决定该层是否重要
            # 初始化使得中间层更重要（先验知识）
            init_gates = torch.ones(num_layers) * 0.5
            # 给中间层更高的初始权重
            mid = num_layers // 2
            for i in range(num_layers):
                init_gates[i] = 0.5 + 0.3 * torch.exp(-torch.tensor((i - mid) ** 2 / (num_layers / 3) ** 2))
            self.layer_importance = nn.Parameter(init_gates)
            print(f"[AdaptivePrefix] 初始化静态层门控: {init_gates}")
        
        if use_dynamic_intensity:
            # 动态强度预测器 - 根据输入时序特征预测每层强度
            self.intensity_predictor = nn.Sequential(
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                self.activation,
                nn.LayerNorm(mlp_hidden_dim // 2),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim // 2, num_layers)
            )
            print(f"[AdaptivePrefix] 启用动态强度预测")
        
        # Prefix生成器（与原版相同）
        final_out_dim = num_layers * prefix_length * hidden_dim
        self.prefix_generator_k = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            self.activation,
            nn.LayerNorm(mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, final_out_dim)
        )
        self.prefix_generator_v = nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            self.activation,
            nn.LayerNorm(mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, final_out_dim)
        )
        
        # Zero初始化（如果需要）
        if init_with_zeros:
            self.apply(self._init_weights_zero)
    
    def _init_weights_zero(self, module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.zeros_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_gates_info(self):
        """获取当前门控状态信息（用于日志和可视化）"""
        info = {}
        if self.use_static_gating:
            gates = torch.sigmoid(self.layer_importance).detach().cpu()
            info['static_gates'] = gates.tolist()
            info['active_layers'] = (gates > 0.5).sum().item()
            info['mean_gate'] = gates.mean().item()
        return info
    
    def forward(self, tokens, attention_mask=None, patch_pooled=None, patch_stats=None):
        batch_size = tokens.shape[0]
        
        # Token编码（与原版相同）
        token_embeds = self.token_embedding(tokens)
        positions = torch.arange(self.max_length, device=tokens.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        pos_embeds = pos_embeds.expand(batch_size, -1, -1)
        combined_embeds = token_embeds + pos_embeds
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            combined_embeds = combined_embeds * mask_expanded
        
        # Token处理
        processed_tokens = self.token_processor(combined_embeds)
        pooled = self.pooling(processed_tokens.permute(0, 2, 1)).squeeze(-1)
        
        # ===== 自适应门控计算 =====
        combined_gates = torch.ones(batch_size, self.num_layers, device=tokens.device)
        
        if self.use_static_gating:
            # 静态门控: sigmoid(重要性 / 温度) + 最小值保护
            static_gates = torch.sigmoid(self.layer_importance / self.temperature)
            static_gates = static_gates.clamp(min=self.min_gate_value)
            combined_gates = combined_gates * static_gates.unsqueeze(0)
        
        if self.use_dynamic_intensity:
            # 动态强度: 根据输入预测
            dynamic_intensity = self.intensity_predictor(pooled)  # [batch, num_layers]
            dynamic_intensity = torch.sigmoid(dynamic_intensity)
            combined_gates = combined_gates * dynamic_intensity
        
        # 生成基础Prefix
        prefix_flat_k = self.prefix_generator_k(pooled)
        prefix_flat_v = self.prefix_generator_v(pooled)
        
        prefix_k = prefix_flat_k.view(batch_size, self.num_layers, self.prefix_length, self.hidden_dim)
        prefix_v = prefix_flat_v.view(batch_size, self.num_layers, self.prefix_length, self.hidden_dim)
        
        # 应用门控: [batch, num_layers, 1, 1] 广播到 prefix
        gate_weights = combined_gates.view(batch_size, self.num_layers, 1, 1)
        prefix_k = prefix_k * gate_weights
        prefix_v = prefix_v * gate_weights
        
        # 组合成 [B, L, 2, prefix_len, H] 格式
        prefix = torch.stack([prefix_k, prefix_v], dim=2)
        
        return prefix


class SparseTokenBasedPrefix(AdaptiveTokenBasedPrefix):
    """
    稀疏版自适应Prefix - 训练时学习稀疏门控，推理时只注入高权重层
    
    进一步优化计算效率，适合部署场景
    """
    def __init__(self, *args, sparsity_target=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity_target = sparsity_target
    
    def forward(self, tokens, attention_mask=None, patch_pooled=None, patch_stats=None):
        # 调用父类forward
        prefix = super().forward(tokens, attention_mask, patch_pooled, patch_stats)
        
        # 训练时增加L0正则化（通过hook实现）
        if self.training and self.use_static_gating:
            # 计算当前稀疏度（有多少层的门控<0.5）
            gates = torch.sigmoid(self.layer_importance / self.temperature)
            current_sparsity = (gates < 0.3).float().mean()
            
            # 存储到buffer用于optimizer外部调整
            self._current_sparsity = current_sparsity.item()
        
        return prefix


