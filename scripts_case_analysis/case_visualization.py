#!/usr/bin/env python
"""
AutoTimes ETTh1 Case Analysis Visualization
==========================================
用于生成预测Case的可视化对比图

使用方法:
    python case_visualization.py --horizon 96 --case_idx 0
    python case_visualization.py --horizon 96 192 336 720 --case_idx 0
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def get_setting(
    model_id,
    model_name,
    data,
    seq_len,
    label_len,
    token_len,
    learning_rate,
    batch_size,
    weight_decay,
    mlp_hidden_dim,
    mlp_hidden_layers,
    cosine,
    mix_embeds,
    prefix_length,
    prefix_mlp_hidden,
    prefix_mlp_layers,
    max_prompt_length,
    des,
):
    """生成与run.py一致的setting名称"""
    return f"{model_id}_{model_name}_{data}_sl{seq_len}_ll{label_len}_tl{token_len}_lr{learning_rate}_bt{batch_size}_wd{weight_decay}_hd{mlp_hidden_dim}_hl{mlp_hidden_layers}_cos{cosine}_mix{mix_embeds}_pl{prefix_length}_pmh{prefix_mlp_hidden}_pml{prefix_mlp_layers}_mpl{max_prompt_length}_{des}_0"


def load_test_data(args, test_pred_len):
    """加载测试数据和模型预测"""
    # 临时修改args进行测试
    args.test_pred_len = test_pred_len
    args.test_seq_len = args.seq_len
    args.test_label_len = args.label_len

    # 获取数据
    exp = Exp_Long_Term_Forecast(args)

    # 加载测试集
    test_data, test_loader = data_provider(args, flag="test")

    return exp, test_data, test_loader


def get_predictions(exp, test_loader, device):
    """获取模型预测结果"""
    predictions = []
    ground_truths = []

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # decoder input
            decoder_input = torch.cat(
                [batch_x, batch_y[:, : exp.args.label_len, :]], dim=1
            )

            # prediction
            outputs = exp.model(batch_x, decoder_input)

            predictions.append(outputs.cpu().numpy())
            ground_truths.append(batch_y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    return predictions, ground_truths


def visualize_case(true, pred, case_idx, horizon, save_dir, dataset_std, dataset_mean):
    """可视化单个case"""
    # 反标准化
    true = true * dataset_std + dataset_mean
    pred = pred * dataset_std + dataset_mean

    # 获取label部分（即预测部分）
    seq_len = len(true) - horizon
    true_future = true[seq_len:]
    pred_future = pred[seq_len:]

    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制完整序列
    x_full = np.arange(len(true))
    x_future = np.arange(seq_len, len(true))

    ax.plot(x_full, true, label="Ground Truth", color="blue", linewidth=1.5, alpha=0.8)
    ax.plot(x_full, pred, label="Prediction", color="red", linewidth=1.5, alpha=0.8)

    # 强调预测部分
    ax.plot(x_future, true_future, label="GT (Future)", color="blue", linewidth=2)
    ax.plot(x_future, pred_future, label="Pred (Future)", color="orange", linewidth=2)

    # 垂直线标记预测起点
    ax.axvline(
        x=seq_len,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Prediction Start",
    )

    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(
        f"ETTh1 Case Analysis - Horizon {horizon} (Case #{case_idx})", fontsize=12
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 计算误差
    mae = np.mean(np.abs(true_future - pred_future))
    mse = np.mean((true_future - pred_future) ** 2)
    rmse = np.sqrt(mse)

    ax.text(
        0.98,
        0.02,
        f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"case_{horizon}_idx{case_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    return mae, rmse


def visualize_multi_case(
    true, pred, case_idx, horizon, save_dir, dataset_std, dataset_mean, n_points=500
):
    """可视化多个时间点"""
    # 反标准化
    true = true * dataset_std + dataset_mean
    pred = pred * dataset_std + dataset_mean

    horizon_len = horizon

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 子图1: 完整序列
    ax1 = axes[0]
    seq_len = len(true) - horizon_len
    x_full = np.arange(len(true))
    x_future = np.arange(seq_len, len(true))

    ax1.plot(x_full, true, label="Ground Truth", color="blue", linewidth=1.2)
    ax1.plot(x_full, pred, label="Prediction", color="red", linewidth=1.2, alpha=0.7)
    ax1.axvline(x=seq_len, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_title(f"Full Sequence - Horizon {horizon}", fontsize=11)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 子图2: 预测部分放大
    ax2 = axes[1]
    ax2.plot(
        x_future, true[seq_len:], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax2.plot(
        x_future, pred[seq_len:], label="Prediction", color="orange", linewidth=1.5
    )
    ax2.fill_between(x_future, true[seq_len:], pred[seq_len:], alpha=0.3, color="gray")
    ax2.set_title(f"Prediction Period (Zoomed)", fontsize=11)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # 计算误差
    true_future = true[seq_len:]
    pred_future = pred[seq_len:]
    mae = np.mean(np.abs(true_future - pred_future))
    rmse = np.sqrt(np.mean((true_future - pred_future) ** 2))

    fig.suptitle(
        f"ETTh1 Case Analysis - Horizon {horizon} (Case #{case_idx}) | MAE: {mae:.4f} RMSE: {rmse:.4f}",
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"case_{horizon}_idx{case_idx}_multi.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    return mae, rmse


def generate_case_analysis(args, horizons=[96, 192, 336, 720], case_idx=0, num_cases=5):
    """生成多个case的分析"""

    # 创建输出目录
    save_dir = f"./case_analysis/{args.model_id}"
    os.makedirs(save_dir, exist_ok=True)

    # 获取数据统计信息
    train_data, _ = data_provider(args, flag="train")
    dataset_std = train_data.std
    dataset_mean = train_data.mean

    print(f"Dataset statistics - Mean: {dataset_mean:.4f}, Std: {dataset_std:.4f}")
    print(f"Generating case analysis for horizons: {horizons}")
    print(f"Output directory: {save_dir}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    results_summary = []

    for horizon in horizons:
        print(f"\n--- Horizon {horizon} ---")

        args.test_pred_len = horizon
        args.test_seq_len = args.seq_len
        args.test_label_len = args.label_len

        exp = Exp_Long_Term_Forecast(args)
        _, test_loader = data_provider(args, flag="test")

        # 获取一个batch的预测
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        decoder_input = torch.cat([batch_x, batch_y[:, : args.label_len, :]], dim=1)

        with torch.no_grad():
            outputs = exp.model(batch_x, decoder_input)

        # 提取指定case
        true = batch_y[case_idx : case_idx + 1].cpu().numpy()[0]
        pred = outputs[case_idx : case_idx + 1].cpu().numpy()[0]

        # 可视化
        mae, rmse = visualize_multi_case(
            true[0], pred[0], case_idx, horizon, save_dir, dataset_std, dataset_mean
        )

        results_summary.append(
            {"horizon": horizon, "case_idx": case_idx, "mae": mae, "rmse": rmse}
        )

        print(f"Horizon {horizon} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # 保存摘要
    summary_path = os.path.join(save_dir, "case_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Case Analysis Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: ETTh1\n")
        f.write(f"Case Index: {case_idx}\n")
        f.write("=" * 40 + "\n\n")
        for r in results_summary:
            f.write(
                f"Horizon {r['horizon']}: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}\n"
            )

    print(f"\nSummary saved to: {summary_path}")
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="AutoTimes Case Visualization")

    # 模型和数据参数
    parser.add_argument("--model", type=str, default="AutoTimes_Gpt2")
    parser.add_argument("--model_id", type=str, default="ETTh1_adaptive_prefix")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--gpu", type=int, default=0)

    # 序列参数
    parser.add_argument("--seq_len", type=int, default=672)
    parser.add_argument("--label_len", type=int, default=576)
    parser.add_argument("--token_len", type=int, default=96)

    # Prefix参数
    parser.add_argument("--prefix_length", type=int, default=4)
    parser.add_argument("--prefix_mlp_hidden", type=int, default=512)
    parser.add_argument("--prefix_mlp_layers", type=int, default=2)
    parser.add_argument("--use_prefix", action="store_true", default=True)
    parser.add_argument("--prefix_injection_mode", type=str, default="deep")
    parser.add_argument("--freeze_llm_except_prefix", action="store_true", default=True)

    # 自适应Prefix参数
    parser.add_argument("--use_adaptive_prefix", action="store_true", default=True)
    parser.add_argument(
        "--adaptive_use_static_gating", action="store_true", default=True
    )
    parser.add_argument(
        "--adaptive_use_dynamic_intensity", action="store_true", default=True
    )

    # 分析参数
    parser.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=[96, 192, 336, 720],
        help="Prediction horizons to analyze",
    )
    parser.add_argument(
        "--case_idx",
        type=int,
        default=0,
        help="Case index to visualize (within a batch)",
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--llm_ckp_dir", type=str, default="/path/to/gpt2")

    args = parser.parse_args()

    # 生成case分析
    results = generate_case_analysis(
        args, horizons=args.horizon, case_idx=args.case_idx
    )

    print("\nCase analysis completed!")
    for r in results:
        print(f"  Horizon {r['horizon']}: MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}")


if __name__ == "__main__":
    main()
