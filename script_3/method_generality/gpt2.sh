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

# 定义参数变量（便于修改）
task_name=long_term_forecast
root_path=./dataset/ETT-small/
data_path=ETTh1.csv
model_id=ETTh1_672_96_adaptive_prefix
data=ETTh1
seq_len=672
label_len=576
token_len=96
test_seq_len=672
test_label_len=576
batch_size=128
learning_rate=0.002
weight_decay=0
mlp_hidden_layers=2
itr=0  # 注意：这里用0，因为run.py中ii从0开始
train_epochs=10
llm_ckp_dir=/home/u4_3090_4/baseModel
gpu=0
des=Gpt2_adaptive
cosine=True
tmax=10
mlp_hidden_dim=512
mix_embeds=True

# 创建必要的目录
mkdir -p ./checkpoints
mkdir -p ./logs

# 动态生成setting路径
setting="${task_name}_${model_id}_${model_name}_${data}_sl${seq_len}_ll${label_len}_tl${token_len}_lr${learning_rate}_bt${batch_size}_wd${weight_decay}_hd${mlp_hidden_dim}_hl${mlp_hidden_layers}_cos${cosine}_mix${mix_embeds}_${des}_${itr}"

# 训练阶段
python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data $data \
  --seq_len $seq_len \
  --label_len $label_len \
  --token_len $token_len \
  --test_seq_len $test_seq_len \
  --test_label_len $test_label_len \
  --test_pred_len 96 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1 \
  --train_epochs $train_epochs \
  --use_amp \
  --llm_ckp_dir $llm_ckp_dir \
  --gpu $gpu \
  --des $des \
  --cosine \
  --tmax $tmax \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mix_embeds \
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

# 测试不同预测长度
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name $task_name \
  --is_training 0 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data $data \
  --seq_len $seq_len \
  --label_len $label_len \
  --token_len $token_len \
  --test_seq_len $test_seq_len \
  --test_label_len $test_label_len \
  --test_pred_len $test_pred_len \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1 \
  --train_epochs $train_epochs \
  --use_amp \
  --llm_ckp_dir $llm_ckp_dir \
  --gpu $gpu \
  --des $des \
  --cosine \
  --tmax $tmax \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mix_embeds \
  --test_dir $setting \
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
