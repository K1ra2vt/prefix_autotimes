import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.mlp import MLP
from .prefix import TokenBasedConditionalPrefix


class PrefixAttentionWrapper(nn.Module):
    def __init__(self, original_attn, layer_idx):
        super(PrefixAttentionWrapper, self).__init__()
        self.original_attn = original_attn
        self.layer_idx = layer_idx
        self.prefix_k = None
        self.prefix_v = None

    def set_prefix_kv(self, prefix_kv):
        if prefix_kv is not None:
            # prefix_kv: [2, batch_size, prefix_length, hidden_dim] where 0=K,1=V
            self.prefix_k, self.prefix_v = prefix_kv[0], prefix_kv[1]
        else:
            self.prefix_k = None
            self.prefix_v = None

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, past_key_values=None, past_key_value=None, cache_position=None, **kwargs):
        qkv = self.original_attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.original_attn.embed_dim, dim=2)
        batch_size, seq_len = query.size()[:2]
        query = query.view(batch_size, seq_len, self.original_attn.num_heads, self.original_attn.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.original_attn.num_heads, self.original_attn.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.original_attn.num_heads, self.original_attn.head_dim).transpose(1, 2)

        if self.prefix_k is not None and self.prefix_v is not None:
            prefix_k = self.prefix_k  # expected [batch, prefix_len, hidden_dim]
            prefix_v = self.prefix_v
            prefix_length = prefix_k.size(1)
            # reshape prefix to heads
            prefix_k_heads = prefix_k.view(batch_size, prefix_length, self.original_attn.num_heads, self.original_attn.head_dim).transpose(1, 2)
            prefix_v_heads = prefix_v.view(batch_size, prefix_length, self.original_attn.num_heads, self.original_attn.head_dim).transpose(1, 2)
            key = torch.cat([prefix_k_heads, key], dim=-2)
            value = torch.cat([prefix_v_heads, value], dim=-2)
            if attention_mask is not None:
                bsz, seqlen = attention_mask.size()
                prefix_mask = torch.ones(bsz, prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.original_attn.head_dim ** 0.5)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        if head_mask is not None:
            attn_scores = attn_scores * head_mask
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        if hasattr(self.original_attn, 'attn_dropout'):
            attn_weights = self.original_attn.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        batch_size, num_heads, seq_len_out, head_dim = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_out, num_heads * head_dim)
        attn_output = self.original_attn.c_proj(attn_output)
        attn_output = self.original_attn.resid_dropout(attn_output)
        outputs = (attn_output, attn_weights if output_attentions else attn_output)
        return outputs


class GPT2PrefixWrapper(nn.Module):
    def __init__(self, gpt2_model):
        super(GPT2PrefixWrapper, self).__init__()
        self.gpt2 = gpt2_model
        self.config = gpt2_model.config
        for i, layer in enumerate(self.gpt2.h):
            layer.attn = PrefixAttentionWrapper(layer.attn, layer_idx=i)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, prefix_kv=None, **kwargs):
        if prefix_kv is not None:
            # prefix_kv: [num_layers, 2, batch_size, prefix_length, hidden_dim]
            for i, layer in enumerate(self.gpt2.h):
                layer.attn.set_prefix_kv(prefix_kv[i])
        outputs = self.gpt2(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return outputs


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.args = configs
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        self.base_gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir, local_files_only=True)
        self.hidden_dim_of_gpt2 = self.base_gpt2.config.hidden_size
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        self.use_prefix = getattr(configs, 'use_prefix', False)
        self.prefix_injection_mode = getattr(configs, 'prefix_injection_mode', 'deep')
        if self.use_prefix:
            from transformers import GPT2Tokenizer
            try:
                tokenizer_path = getattr(configs, 'llm_ckp_dir', '/home/u4_3090_4/baseModel_gpt2')
                temp_tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                vocab_size = temp_tokenizer.vocab_size
                max_length = getattr(configs, 'max_prompt_length', 128)
            except:
                vocab_size = 50257
                max_length = 128
            # 根据配置选择使用自适应prefix还是原版
            use_adaptive_prefix = getattr(configs, 'use_adaptive_prefix', False)
            
            if use_adaptive_prefix:
                # 使用自适应选择性注入的prefix
                from .prefix import AdaptiveTokenBasedPrefix
                self.prefix = AdaptiveTokenBasedPrefix(
                    vocab_size=vocab_size,
                    max_length=max_length,
                    hidden_dim=self.hidden_dim_of_gpt2,
                    prefix_length=configs.prefix_length,
                    num_layers=self.base_gpt2.config.n_layer,
                    mlp_hidden_dim=configs.prefix_mlp_hidden,
                    mlp_layers=configs.prefix_mlp_layers,
                    dropout=getattr(configs, 'dropout', 0.1),
                    init_with_zeros=getattr(configs, 'prefix_init_with_zeros', False),
                    use_static_gating=getattr(configs, 'adaptive_use_static_gating', True),
                    use_dynamic_intensity=getattr(configs, 'adaptive_use_dynamic_intensity', True),
                    min_gate_value=getattr(configs, 'adaptive_min_gate_value', 0.0),
                    temperature=getattr(configs, 'adaptive_gate_temperature', 0.1)
                )
                print(f"[Model] 使用 AdaptiveTokenBasedPrefix (自适应选择性注入)")
                print(f"       静态门控: {getattr(configs, 'adaptive_use_static_gating', True)}")
                print(f"       动态强度: {getattr(configs, 'adaptive_use_dynamic_intensity', True)}")
            else:
                # 使用原版prefix
                self.prefix = TokenBasedConditionalPrefix(
                    vocab_size=vocab_size,
                    max_length=max_length,
                    hidden_dim=self.hidden_dim_of_gpt2,
                    prefix_length=configs.prefix_length,
                    num_layers=self.base_gpt2.config.n_layer,
                    mlp_hidden_dim=configs.prefix_mlp_hidden,
                    mlp_layers=configs.prefix_mlp_layers,
                    dropout=getattr(configs, 'dropout', 0.1),
                    init_with_zeros=getattr(configs, 'prefix_init_with_zeros', False)
                )
                print(f"[Model] 使用 TokenBasedConditionalPrefix (标准版本)")
            if self.prefix_injection_mode == 'deep':
                self.gpt2 = GPT2PrefixWrapper(self.base_gpt2)
            else:
                self.gpt2 = self.base_gpt2
        else:
            self.prefix = None
            self.gpt2 = self.base_gpt2

        mlp_hidden_layers = getattr(configs, 'mlp_hidden_layers', 2)
        if mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2,
                            getattr(configs, 'mlp_hidden_dim', 256), mlp_hidden_layers,
                            getattr(configs, 'dropout', 0.1), getattr(configs, 'mlp_activation', 'tanh'))
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                            getattr(configs, 'mlp_hidden_dim', 256), mlp_hidden_layers,
                            getattr(configs, 'dropout', 0.1), getattr(configs, 'mlp_activation', 'tanh'))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False
        if self.use_prefix:
            for param in self.prefix.parameters():
                param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        if hasattr(self, 'add_scale'):
            self.add_scale.requires_grad = True

        self.prefix_patch_proj = nn.Linear(self.hidden_dim_of_gpt2, getattr(configs, 'prefix_mlp_hidden', 512))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, context_features=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        bs, _, n_vars = x_enc.shape
        batch_size = bs * n_vars
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        times_embeds = self.encoder(fold_out)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            if x_mark_enc.shape[-1] != times_embeds.shape[-1]:
                if x_mark_enc.shape[-1] > times_embeds.shape[-1]:
                    x_mark_enc = x_mark_enc[..., :times_embeds.shape[-1]]
                else:
                    padding = torch.zeros(*x_mark_enc.shape[:-1], times_embeds.shape[-1] - x_mark_enc.shape[-1], device=x_mark_enc.device)
                    x_mark_enc = torch.cat([x_mark_enc, padding], dim=-1)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        if self.use_prefix and self.prefix is not None:
            if context_features is not None and isinstance(context_features, dict) and 'tokens' in context_features:
                tokens = context_features['tokens'].squeeze(1)  # [bs, max_len]
                attention_mask = context_features.get('attention_mask', None)
                if attention_mask is not None:
                    # attention_mask may have shape [bs, 1, max_len] from DataLoader; squeeze that extra dim
                    if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
                        attention_mask = attention_mask.squeeze(1)  # -> [bs, max_len]
                    elif attention_mask.dim() > 2:
                        # fallback: flatten leading dims to [bs, -1] and take last max_length
                        attention_mask = attention_mask.reshape(attention_mask.size(0), -1)
                        if attention_mask.size(1) > self.prefix.max_length:
                            attention_mask = attention_mask[:, -self.prefix.max_length:]
                        else:
                            # pad with ones if shorter
                            pad_len = self.prefix.max_length - attention_mask.size(1)
                            if pad_len > 0:
                                pad = torch.ones(attention_mask.size(0), pad_len, device=attention_mask.device, dtype=attention_mask.dtype)
                                attention_mask = torch.cat([attention_mask, pad], dim=1)
                prefix_vectors = self.prefix(tokens, attention_mask, patch_pooled=None)
            else:
                max_length = getattr(self.prefix, 'max_length', 128)
                tokens = torch.zeros(batch_size, max_length, dtype=torch.long, device=times_embeds.device)
                attention_mask = torch.ones(batch_size, max_length, dtype=torch.long, device=times_embeds.device)
                prefix_vectors = self.prefix(tokens, attention_mask, patch_pooled=None)

            if self.prefix_injection_mode == 'deep':
                num_layers = prefix_vectors.size(1)
                prefix_kv = []
                for layer_idx in range(num_layers):
                    prefix_k = prefix_vectors[:, layer_idx, 0, :, :]
                    prefix_v = prefix_vectors[:, layer_idx, 1, :, :]
                    layer_kv = torch.stack([prefix_k, prefix_v], dim=0)
                    prefix_kv.append(layer_kv)
                prefix_kv = torch.stack(prefix_kv, dim=0)
                outputs = self.gpt2(inputs_embeds=times_embeds, prefix_kv=prefix_kv).last_hidden_state
            else:
                prefix_embeds = prefix_vectors[:, 0, 0, :, :]
                combined_embeds = torch.cat([prefix_embeds, times_embeds], dim=1)
                prefix_attn_mask = torch.ones(batch_size, self.prefix.prefix_length, dtype=torch.long, device=times_embeds.device)
                original_attn_mask = torch.ones(batch_size, token_num, dtype=torch.long, device=times_embeds.device)
                combined_attn_mask = torch.cat([prefix_attn_mask, original_attn_mask], dim=1)
                outputs = self.gpt2(inputs_embeds=combined_embeds, attention_mask=combined_attn_mask).last_hidden_state
                outputs = outputs[:, self.prefix.prefix_length:, :]
        else:
            outputs = self.gpt2(inputs_embeds=times_embeds).last_hidden_state

        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, context_features=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, context_features)

