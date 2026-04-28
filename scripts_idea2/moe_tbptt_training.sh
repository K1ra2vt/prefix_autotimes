#!/bin/bash

# MoE Prefix + TBPTT Training Script
# 混合专家前缀 + 分段反向传播训练

model_name=AutoTimes_Gpt2

# 基本任务设置
task_name=long_term_forecast
root_path=../dataset/ETT-small/
data_path=ETTh1.csv
model_id=ETTh1_MoE_TBPTT
data=ETTh1
seq_len=672
label_len=576
token_len=96
test_seq_len=672
test_label_len=576

# TBPTT和MoE设置
use_tbptt=true
rollout_steps=4  # 推荐2-3步
use_moe_prefix=true

# 训练参数
batch_size=1024  # TBPTT需要更大的batch size来稳定训练
learning_rate=0.0001
weight_decay=0.001
train_epochs=20  # TBPTT训练收敛更快

# 不同模块的学习率
prefix_learning_rate=0.00005  # MoE prefix学习率略低
encoder_decoder_learning_rate=0.0005
llm_ckp_dir=/home/u4_3090_4/baseModel_gpt2
gpu=3
itr=1
num_workers=8
patience=5

# Prefix相关参数
des=MoE_TBPTT_Stage1
cosine=true
tmax=15
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

# 创建目录
mkdir -p ./test_checkpoints
mkdir -p ./test_logs

echo "=========================================="
echo "MoE Prefix + TBPTT 训练"
echo "模型架构: $model_name"
echo "数据集: $data"
echo "序列长度: $seq_len"
echo "预测长度: 96, 192, 336, 720"
echo "TBPTT步数: $rollout_steps"
echo "MoE Prefix: Anchor + Predictor"
echo "训练轮数: $train_epochs"
echo "批次大小: $batch_size"
echo "=========================================="

# 第一阶段训练
echo "开始MoE + TBPTT训练..."
python -u ./run2.py \
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
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --prefix_learning_rate $prefix_learning_rate \
  --encoder_decoder_learning_rate $encoder_decoder_learning_rate \
  --itr $itr \
  --train_epochs $train_epochs \
  --num_workers $num_workers \
  --use_amp \
  --llm_ckp_dir $llm_ckp_dir \
  --gpu $gpu \
  --des $des \
  --cosine \
  --tmax $tmax \
  --mlp_hidden_dim $mlp_hidden_dim \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --mix_embeds \
  --use_tbptt \
  --rollout_steps $rollout_steps \
  --use_moe_prefix \
  --drop_last


echo "第一阶段训练完成，开始测试不同预测长度..."

# 测试不同预测长度
for test_pred_len in 96 192 336 720
do
echo "测试预测长度: $test_pred_len"
python -u ./run2.py \
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
  --prefix_learning_rate $prefix_learning_rate \
  --encoder_decoder_learning_rate $encoder_decoder_learning_rate \
  --itr $itr \
  --train_epochs $train_epochs \
  --num_workers $num_workers \
  --use_amp \
  --llm_ckp_dir $llm_ckp_dir \
  --gpu $gpu \
  --des $des \
  --cosine \
  --tmax $tmax \
  --mlp_hidden_dim $mlp_hidden_dim \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --mix_embeds \
  --use_tbptt \
  --rollout_steps $rollout_steps \
  --use_moe_prefix \
  --visualize \
  --drop_last
done

echo "=========================================="
echo "MoE + TBPTT 训练完成！"
echo "=========================================="
echo "已完成:"
echo "1. TBPTT训练：$rollout_steps 步rollout"
echo "2. MoE Prefix：Anchor + Predictor 双专家"
echo "3. 完美模拟推理过程，消除训练-推理gap"
echo ""
echo "结果保存在 result_long_term_forecast.txt 中"
echo "模型checkpoint保存在 checkpoints/$setting/ 目录下"
echo "=========================================="