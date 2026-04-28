model_name=AutoTimes_Gpt2

# 条件化prefix tuning的核心参数 (减小参数进行测试)
task_name=long_term_forecast
root_path=../dataset/ETT-small/
data_path=ETTh1.csv
model_id=ETTh1_test_conditional_prefix_tuning
data=ETTh1
seq_len=672   # 减小序列长度
label_len=576
token_len=96
test_seq_len=672
test_label_len=576

training_stage=prefix  # 训练阶段

# 训练参数
batch_size=1536  # 减小batch size进行测试
learning_rate=0.0005
weight_decay=0.0001
train_epochs=25

# 不同模块的学习率设置
prefix_learning_rate=0.0001    # Prefix MLP的学习率，通常较小
encoder_decoder_learning_rate=0.001  # Encoder/Decoder的学习率
llm_ckp_dir=/home/u4_3090_4/baseModel_gpt2
gpu=3
itr=1
num_workers=8  # 添加num_workers参数
patience=3  # early stopping patience

# Prefix相关参数 (减小参数进行测试)
des=ConditionalStructuredPrefix
cosine=True
tmax=10
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

# 创建必要的目录
mkdir -p ./test_checkpoints
mkdir -p ./test_logs

echo "=========================================="
echo "条件化结构化提示词Prefix Tuning训练开始"
echo "=========================================="
echo "模型架构: $model_name"
echo "数据集: $data"
echo "序列长度: $seq_len"
echo "预测长度: 96, 192, 336, 720"
echo "Prefix长度: $prefix_length"
echo "结构化提示词最大长度: $max_prompt_length"
echo "训练轮数: $train_epochs"
echo "批次大小: $batch_size"
echo "学习率: $learning_rate"
echo "Prefix学习率: $prefix_learning_rate"
echo "Encoder/Decoder学习率: $encoder_decoder_learning_rate"
echo "只训练Prefix MLP和时序Encoder/Decoder，不训练LLM"
echo "=========================================="

# 训练阶段
echo "开始训练条件化Prefix Tuning模型..."
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
  --use_prefix \
  --prefix_length $prefix_length \
  --prefix_mlp_hidden $prefix_mlp_hidden \
  --prefix_mlp_layers $prefix_mlp_layers \
  --freeze_llm_except_prefix \
  --max_prompt_length $max_prompt_length \
  --prefix_injection_mode $prefix_injection_mode \
  --training_stage $training_stage 

echo "训练完成，开始测试不同预测长度..."

# 测试不同预测长度
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
  --training_stage $training_stage 
done

echo "=========================================="
echo "条件化结构化提示词Prefix Tuning完成！"
echo "=========================================="
echo "实现的功能："
echo "1. 结构化提示词模板 -> Token转换"
echo "2. MLP处理Token序列生成Prefix向量"
echo "3. 条件化Prefix注入到LLM的每一层"
echo "4. 只训练Prefix MLP和时序Encoder/Decoder"
echo "5. LLM参数完全冻结"
echo ""
echo "结果保存在 result_long_term_forecast.txt 中"
echo "模型checkpoint保存在 checkpoints/$setting/ 目录下"
echo "=========================================="