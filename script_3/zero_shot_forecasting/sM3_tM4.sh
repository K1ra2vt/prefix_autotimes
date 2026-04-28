model_name=AutoTimes_Gpt2
root_path=/home/u4_3090_4/dataset/tsf

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
  --is_training 1 \
  --root_path $root_path \
  --data_path m3_yearly_dataset.tsf \
  --test_data_path m4_yearly_dataset.tsf \
  --seasonal_patterns 'Yearly' \
  --model_id m3_Yearly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 6 \
  --label_len 0 \
  --token_len 6 \
  --test_seq_len 6 \
  --test_label_len 0 \
  --test_pred_len 6 \
  --learning_rate 0.0001 \
  --mlp_hidden_dim 256 \
  --mlp_hidden_layers 3 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --cosine \
  --tmax 10 \
  --val_set_shuffle \
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
  --is_training 1 \
  --root_path $root_path \
  --data_path m3_quarterly_dataset.tsf \
  --test_data_path m4_quarterly_dataset.tsf \
  --seasonal_patterns 'Quarterly' \
  --model_id m3_Quarterly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 8 \
  --label_len 0 \
  --token_len 8 \
  --test_seq_len 8 \
  --test_label_len 0 \
  --test_pred_len 8 \
  --learning_rate 0.000005 \
  --mlp_hidden_dim 1024 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --cosine \
  --tmax 10 \
  --val_set_shuffle \
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
  --is_training 1 \
  --root_path $root_path \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_monthly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 24 \
  --label_len 0 \
  --token_len 24 \
  --test_seq_len 24 \
  --test_label_len 0 \
  --test_pred_len 24 \
  --learning_rate 0.00001 \
  --mlp_hidden_dim 512 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --cosine \
  --tmax 10 \
  --val_set_shuffle \
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
  --is_training 1 \
  --root_path $root_path \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_weekly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 26 \
  --label_len 13 \
  --token_len 13 \
  --test_seq_len 26 \
  --test_label_len 13 \
  --test_pred_len 13 \
  --learning_rate 0.001 \
  --mlp_hidden_dim 256 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --cosine \
  --tmax 10 \
  --val_set_shuffle \
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
  --is_training 1 \
  --root_path $root_path \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_daily_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 28 \
  --label_len 14 \
  --token_len 14 \
  --test_seq_len 28 \
  --test_label_len 14 \
  --test_pred_len 14 \
  --learning_rate 0.0001 \
  --mlp_hidden_dim 256 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --cosine \
  --tmax 10 \
  --val_set_shuffle \
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
  --is_training 1 \
  --root_path $root_path \
  --data_path m3_monthly_dataset.tsf \
  --test_data_path m4_hourly_dataset.tsf \
  --seasonal_patterns 'Monthly' \
  --model_id m3_Monthly_adaptive_prefix \
  --model $model_name \
  --data tsf \
  --seq_len 48 \
  --label_len 24 \
  --token_len 24 \
  --test_seq_len 48 \
  --test_label_len 24 \
  --test_pred_len 48 \
  --learning_rate 0.001 \
  --mlp_hidden_dim 128 \
  --mlp_hidden_layers 3 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --cosine \
  --tmax 10 \
  --val_set_shuffle \
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
