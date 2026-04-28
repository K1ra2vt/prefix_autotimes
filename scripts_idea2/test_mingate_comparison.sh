#!/bin/bash
# 参数对比测试脚本 - 最小门控值对比

model_name=AutoTimes_Gpt2
task_name=long_term_forecast
root_path=../dataset/ETT-small/
data_path=ETTh1.csv
data=ETTh1
seq_len=672
label_len=576
token_len=96
test_seq_len=672
test_label_len=576

# 训练参数
batch_size=1536
learning_rate=0.0001
weight_decay=0 
train_epochs=10
prefix_learning_rate=0.0001
encoder_decoder_learning_rate=0.0005
llm_ckp_dir=/home/u4_3090_4/baseModel_gpt2
gpu=0
num_workers=10

# Prefix基础参数
prefix_length=4
prefix_mlp_hidden=512
prefix_mlp_layers=2
max_prompt_length=64
mlp_hidden_dim=512
mlp_hidden_layers=2

# 测试不同的最小门控值（防止完全关闭的保护阈值）
min_gates=(0.0 0.05 0.1 0.2 0.3)

echo "=========================================="
echo "开始最小门控值对比测试"
echo "测试最小门控值: ${min_gates[@]}"
echo "=========================================="

for min_gate in "${min_gates[@]}"
do
    model_id="ETTh1_mingate_${min_gate}"
    des="mg${min_gate}"
    
    echo ""
    echo "=========================================="
    echo "测试最小门控值: $min_gate"
    echo "模型ID: $model_id"
    echo "=========================================="
    
    # 训练
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
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --prefix_learning_rate $prefix_learning_rate \
      --encoder_decoder_learning_rate $encoder_decoder_learning_rate \
      --itr 1 \
      --train_epochs $train_epochs \
      --num_workers $num_workers \
      --use_amp \
      --llm_ckp_dir $llm_ckp_dir \
      --gpu $gpu \
      --des $des \
      --cosine \
      --tmax 25 \
      --mlp_hidden_dim $mlp_hidden_dim \
      --mlp_hidden_layers $mlp_hidden_layers \
      --use_prefix \
      --prefix_length $prefix_length \
      --prefix_mlp_hidden $prefix_mlp_hidden \
      --prefix_mlp_layers $prefix_mlp_layers \
      --freeze_llm_except_prefix \
      --max_prompt_length $max_prompt_length \
      --prefix_injection_mode deep \
      --use_adaptive_prefix \
      --adaptive_use_static_gating \
      --adaptive_use_dynamic_intensity \
      --adaptive_min_gate_value $min_gate \
      --adaptive_gate_temperature 0.1
    
    # 测试
    echo "测试预测长度: 96, 192, 336, 720"
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
          --prefix_learning_rate $prefix_learning_rate \
          --itr 1 \
          --train_epochs $train_epochs \
          --use_amp \
          --llm_ckp_dir $llm_ckp_dir \
          --gpu $gpu \
          --des $des \
          --cosine \
          --tmax 25 \
          --mlp_hidden_dim $mlp_hidden_dim \
          --mlp_hidden_layers $mlp_hidden_layers \
          --use_prefix \
          --prefix_length $prefix_length \
          --prefix_mlp_hidden $prefix_mlp_hidden \
          --prefix_mlp_layers $prefix_mlp_layers \
          --freeze_llm_except_prefix \
          --max_prompt_length $max_prompt_length \
          --prefix_injection_mode deep \
          --use_adaptive_prefix \
          --adaptive_use_static_gating \
          --adaptive_use_dynamic_intensity \
          --adaptive_min_gate_value $min_gate \
          --adaptive_gate_temperature 0.1
    done
    
    echo "最小门控值 $min_gate 测试完成"
done

echo ""
echo "=========================================="
echo "所有最小门控值测试完成！"
echo "=========================================="
