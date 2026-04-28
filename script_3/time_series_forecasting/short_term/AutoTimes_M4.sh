export CUDA_VISIBLE_DEVICES=1

model_name=AutoTimes_Gpt2

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

root_path=/home/u4_3090_4/dataset/m4

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly_adaptive_prefix \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --cosine \
  --tmax 10 \
  --weight_decay 0.00001 \
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
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly_adaptive_prefix \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --learning_rate 0.00005 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --cosine \
  --tmax 10 \
  --weight_decay 0.000005 \
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
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly_adaptive_prefix \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --learning_rate 0.00005 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --cosine \
  --tmax 10 \
  --weight_decay 0.000001 \
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
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly_adaptive_prefix \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --cosine \
  --tmax 10 \
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
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily_adaptive_prefix \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --learning_rate 0.0005 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --weight_decay 0.000005 \
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
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly_adaptive_prefix \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp_adaptive' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --cosine \
  --tmax 10 \
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
