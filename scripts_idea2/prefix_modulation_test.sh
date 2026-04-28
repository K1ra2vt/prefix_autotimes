model_name=AutoTimes_Gpt2

task_name=long_term_forecast
root_path=../dataset/ETT-small/
data_path=ETTh1.csv
model_id=ETTh1_prompt_modulation
data=ETTh1
seq_len=672
label_len=576
token_len=96
test_seq_len=672
test_label_len=576

# 训练参数 - 第一阶段重点训练prefix
batch_size=1536
learning_rate=0.0001
weight_decay=0 
train_epochs=10

# 不同模块的学习率设置
prefix_learning_rate=0.0001    # Prefix MLP的学习率
encoder_decoder_learning_rate=0.0005  # Encoder/Decoder的学习率
llm_ckp_dir=/home/u4_3090_4/baseModel_gpt2
gpu=3
itr=1
num_workers=10
patience=3

# Prefix相关参数
des=new_Prompt_modulation
cosine=True
tmax=25
mlp_hidden_dim=512
mlp_hidden_layers=2

mix_embeds=false 
max_prompt_length=64
use_prefix=true # 只启用这个使用的是基于TokenBased的preifx mlp
prefix_length=4
prefix_mlp_hidden=512
prefix_mlp_layers=2
freeze_llm_except_prefix=true
prefix_injection_mode=deep

# 新增：调制相关参数
use_modulation=true
modulation_dim=512
modulation_gate_dim=512

echo "=========================================="
echo "测试Prompt-Gated领域自适应对齐机制"
echo "模型名称: $model_name"
echo "数据集: $data"
echo "序列长度: $seq_len"
echo "Prefix长度: $prefix_length"
echo "调制启用: $use_modulation"
echo "调制维度: $modulation_dim"
echo "训练轮数: $train_epochs"
echo "批次大小: $batch_size"
echo "=========================================="

echo "开始快速训练..."
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
  --mlp_hidden_layers $mlp_hidden_layers \
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --prefix_injection_mode $prefix_injection_mode \
  --use_modulation \
  --modulation_dim $modulation_dim \
  --modulation_gate_dim $modulation_gate_dim

echo "训练完成,开始测试"
for test_pred_len in 96 192 336 720
do
echo "测试预测长度: $test_pred_len"
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
  --use_modulation \
  --modulation_dim $modulation_dim \
  --modulation_gate_dim $modulation_gate_dim 
  # --visualize 
done
#  --test_dir "long_term_forecast_ETTh1_prompt_modulation_AutoTimes_Gpt2_ETTh1_sl672_ll576_tl96_lr0.0005_bt1536_wd0_hd512_hl2_cosTrue_mixFalse_pl4_pmh512_pml2_mpl64_PromptModulation_0" \
echo "脚本运行完毕"