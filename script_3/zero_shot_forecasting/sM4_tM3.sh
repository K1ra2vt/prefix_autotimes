model_name=AutoTimes_Llama

# Prefix相关参数（保持与adaptive_prefix_test.sh一致）
prefix_learning_rate=0.0001
encoder_decoder_learning_rate=0.0005
prefix_length=4
prefix_mlp_hidden=512
prefix_mlp_layers=2
prefix_injection_mode=deep
freeze_llm_except_prefix=true
max_prompt_length=64

# 自适应选择性注入参数（保持与adaptive_prefix_test.sh一致）
use_adaptive_prefix=true
adaptive_use_static_gating=true
adaptive_use_dynamic_intensity=true
adaptive_min_gate_value=0.1
adaptive_gate_temperature=0.1

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_yearly_dataset.tsf \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 12 \
  --label_len 6 \
  --token_len 6 \
  --test_seq_len 12 \
  --test_label_len 6 \
  --test_pred_len 6 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --test_dir short_term_forecast_m4_Yearly_adaptive_prefix_AutoTimes_Llama_m4_sl12_ll6_tl6_lr0.0001_bt16_wd1e-05_hd512_hl2_cosTrue_mixFalse_Exp_adaptive_0 \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --prefix_injection_mode $prefix_injection_mode \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --use_adaptive_prefix \
  --adaptive_use_static_gating \
  --adaptive_use_dynamic_intensity \
  --adaptive_min_gate_value $adaptive_min_gate_value \
  --adaptive_gate_temperature $adaptive_gate_temperature

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_quarterly_dataset.tsf \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 16 \
  --label_len 8 \
  --token_len 8 \
  --test_seq_len 16 \
  --test_label_len 8 \
  --test_pred_len 8 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --test_dir short_term_forecast_m4_Quarterly_adaptive_prefix_AutoTimes_Llama_m4_sl16_ll8_tl8_lr5e-05_bt16_wd5e-06_hd512_hl2_cosTrue_mixFalse_Exp_adaptive_0 \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --prefix_injection_mode $prefix_injection_mode \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --use_adaptive_prefix \
  --adaptive_use_static_gating \
  --adaptive_use_dynamic_intensity \
  --adaptive_min_gate_value $adaptive_min_gate_value \
  --adaptive_gate_temperature $adaptive_gate_temperature

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_monthly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 36 \
  --label_len 18 \
  --token_len 18 \
  --test_seq_len 36 \
  --test_label_len 18 \
  --test_pred_len 18 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --test_dir short_term_forecast_m4_Monthly_adaptive_prefix_AutoTimes_Llama_m4_sl36_ll18_tl18_lr5e-05_bt16_wd1e-06_hd1024_hl2_cosTrue_mixFalse_Exp_adaptive_0 \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --prefix_injection_mode $prefix_injection_mode \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --use_adaptive_prefix \
  --adaptive_use_static_gating \
  --adaptive_use_dynamic_intensity \
  --adaptive_min_gate_value $adaptive_min_gate_value \
  --adaptive_gate_temperature $adaptive_gate_temperature

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/tsf \
  --test_data_path m3_other_dataset.tsf \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 16 \
  --label_len 8 \
  --token_len 8 \
  --test_seq_len 16 \
  --test_label_len 8 \
  --test_pred_len 8 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --test_dir short_term_forecast_m4_Quarterly_adaptive_prefix_AutoTimes_Llama_m4_sl16_ll8_tl8_lr5e-05_bt16_wd5e-06_hd512_hl2_cosTrue_mixFalse_Exp_adaptive_0 \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --prefix_injection_mode $prefix_injection_mode \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --use_adaptive_prefix \
  --adaptive_use_static_gating \
  --adaptive_use_dynamic_intensity \
  --adaptive_min_gate_value $adaptive_min_gate_value \
  --adaptive_gate_temperature $adaptive_gate_temperature
