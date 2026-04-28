#!/bin/bash

# ============================================
# AutoTimes Adaptive Prefix - ETTh1 Case Analysis
# ============================================
# 请在运行前配置以下参数:
#   - GPU编号
#   - LLM模型路径
#   - 数据集路径
# ============================================

# ====== 需要配置的参数 ======
GPU_ID="0"                    # 修改为可用的GPU编号
LLM_CKP_DIR="/path/to/gpt2"  # 修改为GPT2模型路径
ROOT_PATH="./dataset/ETT-small/"  # 修改为数据集路径

# ====== 模型配置 ======
model_name=AutoTimes_Gpt2
task_name=long_term_forecast
data_path=ETTh1.csv
model_id=ETTh1_adaptive_prefix
data=ETTh1
seq_len=672
label_len=576
token_len=96
test_seq_len=672
test_label_len=576

# ====== 训练参数 ======
batch_size=1536
learning_rate=0.0001
weight_decay=0
train_epochs=10

# ====== Prefix参数 ======
prefix_learning_rate=0.0001
encoder_decoder_learning_rate=0.0005
itr=1
num_workers=10
patience=3

# ====== Prefix配置 ======
des=adaptivePrefix
cosine=True
tmax=25
mlp_hidden_dim=512
mlp_hidden_layers=2

mix_embeds=false 
max_prompt_length=64
use_prefix=true
prefix_length=4
prefix_mlp_hidden=512
prefix_mlp_layers=2
freeze_llm_except_prefix=true
prefix_injection_mode=deep

# ====== 自适应Prefix配置 ======
use_adaptive_prefix=true
adaptive_use_static_gating=true
adaptive_use_dynamic_intensity=true
adaptive_min_gate_value=0.1
adaptive_gate_temperature=0.1

echo "=========================================="
echo "AutoTimes Adaptive Prefix - ETTh1"
echo "模型: $model_name"
echo "数据集: $data"
echo "GPU: $GPU_ID"
echo "序列长度: $seq_len"
echo "Prefix长度: $prefix_length"
echo "训练轮数: $train_epochs"
echo "=========================================="

# ====== 训练阶段 ======
echo "开始训练..."
python -u run.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data $data \
  --seq_len $seq_len \
  --label_len $label_len \
  --token_len $token_len \
  --test_seq_len $test_seq_len \
  --test_label_len $test_label_len \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --prefix_learning_rate $prefix_learning_rate \
  --encoder_decoder_learning_rate $encoder_decoder_learning_rate \
  --itr $itr \
  --train_epochs $train_epochs \
  --num_workers $num_workers \
  --use_amp \
  --llm_ckp_dir $LLM_CKP_DIR \
  --gpu $GPU_ID \
  --des $des \
  --cosine \
  --tmax $tmax \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_hidden_layers $mlp_hidden_layers \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --prefix_injection_mode $prefix_injection_mode \
  --use_adaptive_prefix \
  --adaptive_use_static_gating \
  --adaptive_use_dynamic_intensity \
  --adaptive_min_gate_value $adaptive_min_gate_value \
  --adaptive_gate_temperature $adaptive_gate_temperature

# 保存训练对应的setting名称
TRAINING_SETTING=$(python -c "
args_dict = {
    'task_name': '$task_name',
    'model_id': '$model_id',
    'model': '$model_name',
    'data': '$data',
    'seq_len': $seq_len,
    'label_len': $label_len,
    'token_len': $token_len,
    'learning_rate': $learning_rate,
    'batch_size': $batch_size,
    'weight_decay': $weight_decay,
    'mlp_hidden_dim': $mlp_hidden_dim,
    'mlp_hidden_layers': $mlp_hidden_layers,
    'cosine': $cosine,
    'mix_embeds': $mix_embeds,
    'prefix_length': $prefix_length,
    'prefix_mlp_hidden': $prefix_mlp_hidden,
    'prefix_mlp_layers': $prefix_mlp_layers,
    'max_prompt_length': $max_prompt_length,
    'des': '$des',
    'ii': 0
}
setting = '{task_name}_{model_id}_{model}_{data}_sl{seq_len}_ll{label_len}_tl{token_len}_lr{learning_rate}_bt{batch_size}_wd{weight_decay}_hd{mlp_hidden_dim}_hl{mlp_hidden_layers}_cos{cosine}_mix{mix_embeds}_pl{prefix_length}_pmh{prefix_mlp_hidden}_pml{prefix_mlp_layers}_mpl{max_prompt_length}_{des}_{ii}'.format(**args_dict)
print(setting)
")

echo "训练完成, setting: $TRAINING_SETTING"

# ====== 测试阶段 ======
for test_pred_len in 96 192 336 720
do
  echo "测试预测长度: $test_pred_len"
  python -u run.py \
    --task_name $task_name \
    --is_training 0 \
    --root_path $ROOT_PATH \
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
    --prefix_learning_rate $prefix_learning_rate \
    --itr 1 \
    --train_epochs $train_epochs \
    --use_amp \
    --llm_ckp_dir $LLM_CKP_DIR \
    --gpu $GPU_ID \
    --des $des \
    --cosine \
    --tmax $tmax \
    --mlp_hidden_dim $mlp_hidden_dim \
    --mlp_hidden_layers $mlp_hidden_layers \
    --use_prefix \
    --prefix_length $prefix_length \
    --prefix_mlp_hidden $prefix_mlp_hidden \
    --prefix_mlp_layers $prefix_mlp_layers \
    --freeze_llm_except_prefix \
    --max_prompt_length $max_prompt_length \
    --prefix_injection_mode $prefix_injection_mode \
    --use_adaptive_prefix \
    --adaptive_use_static_gating \
    --adaptive_use_dynamic_intensity \
    --adaptive_min_gate_value $adaptive_min_gate_value \
    --adaptive_gate_temperature $adaptive_gate_temperature
done

echo "脚本运行完毕"