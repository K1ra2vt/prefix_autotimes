import argparse
import os
import torch
from models.Preprocess_Llama import LlamaModel
from models.Preprocess_Gpt2 import Gpt2Model
from data_provider.data_loader import Dataset_Preprocess
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default=None, help='llm checkpoints dir')
    parser.add_argument('--llm_model', type=str, default='llama', choices=['llama', 'gpt2'], help='llm model type')
    parser.add_argument('--dataset', type=str, default='ETTh1', 
                        help='dataset to preprocess, options:[ETTh1, electricity, weather, traffic]')
    args = parser.parse_args()
    print(args.dataset)
    
    # 根据模型类型自动设置checkpoint路径
    if args.llm_ckp_dir is None:
        if args.llm_model == 'llama':
            args.llm_ckp_dir = '/home/u4_3090_4/baseModel_llama' # 先不要使用
        elif args.llm_model == 'gpt2':
            args.llm_ckp_dir = '/home/u4_3090_4/baseModel_gpt2'
    
    # 根据模型类型选择预处理模型
    if args.llm_model == 'llama':
        model = LlamaModel(args)
    elif args.llm_model == 'gpt2':
        model = Gpt2Model(args)
    else:
        raise ValueError(f"Unsupported LLM model: {args.llm_model}")

    seq_len = 672
    label_len = 576
    pred_len = 96
    
    # 根据数据集类型设置路径
    assert args.dataset in ['ETTh1', 'electricity', 'weather', 'traffic']
    if args.dataset == 'ETTh1':
        root_path = '../dataset/ETT-small/'
        data_path = 'ETTh1.csv'
    elif args.dataset == 'electricity':
        root_path = '../dataset/electricity/'
        data_path = 'electricity.csv'
    elif args.dataset == 'weather':
        root_path = '../dataset/weather/'
        data_path = 'weather.csv'
    elif args.dataset == 'traffic':
        root_path = '../dataset/traffic/'
        data_path = 'traffic.csv'
    
    # 创建数据集
    data_set = Dataset_Preprocess(
        root_path=root_path,
        data_path=data_path,
        size=[seq_len, label_len, pred_len])

    data_loader = DataLoader(
        data_set,
        batch_size=128,
        shuffle=False,
    )

    from tqdm import tqdm
    print(len(data_set.data_stamp))
    print(data_set.tot_len)
    
    # 保存路径与数据路径一致
    save_dir_path = root_path
    
    output_list = []
    for idx, data in tqdm(enumerate(data_loader)):
        # data[0] is tuple of strings (batch of text prompts)
        # data[1] is dict of batched tensors (batch of context features)
        seq_x_marks = data[0]  # tuple of strings
        batched_context_features = data[1]  # dict of tensors with shape [batch_size, ...]

        # Convert batched context features back to list of dicts (one per sample)
        batch_size = len(seq_x_marks)
        context_features_list = []
        for i in range(batch_size):
            sample_features = {}
            for key, tensor in batched_context_features.items():
                sample_features[key] = tensor[i]  # Extract i-th sample
            context_features_list.append(sample_features)

        output = model(seq_x_marks, context_features_list)
        output_list.append(output.detach().cpu())
    result = torch.cat(output_list, dim=0)
    print(result.shape)
    
    # 保存到正确位置
    save_path = os.path.join(save_dir_path, f'{args.dataset}_{args.llm_model}.pt')
    torch.save(result, save_path)
    print(f"Preprocessed file saved to: {save_path}")