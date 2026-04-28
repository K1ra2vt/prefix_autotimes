model_name=AutoTimes_Llama
# 使用llama模型在etth1上跑来训练，这个版本不进行prefix

# 核心参数设置
task_name=long_term_forecast
root_path=../dataset/ETT-small/
data_path=ETTh1.csv
model_id=ETTh1_672_96
data=ETTh1
seq_len=672
label_len=576
token_len=96
test_seq_len=672
test_label_len=576

# 训练参数 (为LLaMA调整参数)
batch_size=256  # LLaMA需要更大的内存，使用较小的batch_size
learning_rate=0.0005
weight_decay=0  # 默认weight_decay
train_epochs=10
itr=1  # 从0开始

# LLaMA模型路径
llm_ckp_dir=/home/u4_3090_4/baseModel_llama
gpu=3  # 使用GPU 3
num_workers=0
patience=3

# 模型架构参数
des=Llama_ETTh1
cosine=True
tmax=10
mlp_hidden_dim=256  # 默认值
mlp_hidden_layers=0  # 使用线性层（论文是用的线性层）
mix_embeds=True

echo "=========================================="
echo "LLaMA模型时间序列预测训练开始"
echo "=========================================="
echo "模型架构: $model_name"
echo "数据集: $data"
echo "序列长度: $seq_len"
echo "预测长度: 96, 192, 336, 720"
echo "训练轮数: $train_epochs"
echo "批次大小: $batch_size"
echo "学习率: $learning_rate"
echo "使用GPU: $gpu"
echo "LLaMA模型路径: $llm_ckp_dir"
echo "=========================================="

# 训练阶段
echo "开始训练LLaMA模型..."
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
  --mix_embeds \
  --drop_last

test_dir="${task_name}_${model_id}_${model_name}_${data}_sl${seq_len}_ll${label_len}_tl${token_len}_lr${learning_rate}_bt${batch_size}_wd${weight_decay}_hd${mlp_hidden_dim}_hl${mlp_hidden_layers}_cos${cosine}_mix${mix_embeds}_${des}_${itr}"

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
  --mix_embeds \
  --drop_last \
  --test_dir $test_dir
done


echo "=========================================="
echo "LLaMA模型时间序列预测完成！"
echo "=========================================="
echo "实现的功能："
echo "1. 使用LLaMA作为骨干网络的时间序列预测"
echo "2. 线性编码器/解码器处理时序数据"
echo "3. 支持多尺度预测长度测试"
echo ""
echo "结果保存在 result_long_term_forecast.txt 中"
echo "模型checkpoint保存在 checkpoints/\$setting/ 目录下"
echo "=========================================="