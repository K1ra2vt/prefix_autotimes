model_name=AutoTimes_Opt_1b

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
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96_adaptive_prefix \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 2048 \
  --learning_rate 0.001 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir ./models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 \
  --gpu 0 \
  --des 'Opt1b_adaptive' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 256 \
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

for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96_adaptive_prefix \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 2048 \
  --learning_rate 0.001 \
  --itr 1 \
  --train_epochs 10 \
  --use_amp \
  --llm_ckp_dir ./models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62 \
  --gpu 0 \
  --des 'Opt1b_adaptive' \
  --cosine \
  --tmax 10 \
  --mlp_hidden_dim 256 \
  --test_dir long_term_forecast_ETTh1_672_96_adaptive_prefix_AutoTimes_Opt_1b_ETTh1_sl672_ll576_tl96_lr0.001_bt2048_wd0_hd256_hl2_cosTrue_mixFalse_Opt1b_adaptive_0 \
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
done
