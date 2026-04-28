#!/bin/bash

# ============================================
# AutoTimes Adaptive Prefix - ETTh1 Case Analysis
# ============================================

# ====== GPU配置 ======
# 如果知道GPU编号，直接填写；如果不知道，运行 nvidia-smi 查看
GPU_ID="0"                    # 修改为可用的GPU编号

# ====== 模型配置 ======
# GPT2模型存放目录
LLM_MODEL_DIR="/root/.cache/huggingface/hub/models--gpt2"
# 或指定其他路径
LLM_CKP_DIR="${LLM_MODEL_DIR}/tf_model"  # 根据实际模型结构调整

# ====== 数据集配置 ======
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

# ====== 检查并下载模型 ======
check_and_download_model() {
    echo "检查GPT2模型..."
    
    # 检查模型是否已存在
    if [ -d "$LLM_MODEL_DIR" ]; then
        echo "模型已存在于: $LLM_MODEL_DIR"
    else
        echo "正在下载GPT2模型..."
        python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; \
                  tokenizer = GPT2Tokenizer.from_pretrained('gpt2'); \
                  model = GPT2LMHeadModel.from_pretrained('gpt2'); \
                  print('GPT2模型下载完成')"
        
        # 检查默认缓存目录
        DEFAULT_CACHE=$(python -c "from transformers import TRANSFORMERS_CACHE; print(TRANSFORMERS_CACHE)")
        echo "模型缓存目录: $DEFAULT_CACHE"
    fi
}

# ====== 训练阶段 ======
run_training() {
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
}

# ====== 测试阶段 ======
run_test() {
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
}

# ====== 主流程 ======
echo "步骤1: 检查GPU和模型"
check_and_download_model

echo "步骤2: 开始训练"
run_training

echo "步骤3: 测试"
run_test

echo "脚本运行完毕"