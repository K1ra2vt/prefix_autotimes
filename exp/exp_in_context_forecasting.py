from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import mape_loss, mase_loss, smape_loss, zero_shot_smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# In our in-context learning setting
# the task is to apply a forecaster, trained on a source dataset, to an unseen target dataset
# Additionally, several task demonstrations from the target domain, 
# referred to as time series prompts are available during inference
# Concretely, AutoTimes trains LLMs on the source domain with a larger context length to place the additional time series prompt. 
# See ```Dataset_TSF_ICL``` in ```data_loader.py``` for the construction of time series prompts

warnings.filterwarnings('ignore')

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))

def first_difference_alignment_loss(pred, true, alpha=0.1):
    """
    Loss function combining MSE with first-difference alignment for trend preservation.

    Args:
        pred: Predicted values [batch_size, seq_len, features]
        true: True values [batch_size, seq_len, features]
        alpha: Weight for first-difference term (0.1 = 10% of total loss)

    Returns:
        Combined loss value
    """
    # Main MSE loss
    mse_loss = nn.MSELoss()(pred, true)

    # First difference alignment (trend preservation)
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    true_diff = true[:, 1:, :] - true[:, :-1, :]
    diff_loss = nn.MSELoss()(pred_diff, true_diff)

    # Combine losses
    total_loss = (1 - alpha) * mse_loss + alpha * diff_loss
    return total_loss

class Exp_In_Context_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_In_Context_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        self.device = self.args.gpu
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Enhanced optimizer selection with prefix-specific parameter groups and freezing strategies.
        """
        # Define parameter groups based on prefix mode and freezing strategy
        encoder_params = []
        prefix_params = []
        llm_params = []

        freeze_llm_except_prefix = getattr(self.args, 'freeze_llm_except_prefix', True)
        use_prefix = getattr(self.args, 'use_prefix', False)

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # Categorize parameters
            if 'encoder' in n or 'decoder' in n:
                encoder_params.append(p)
            elif use_prefix and ('prefix' in n):
                prefix_params.append(p)
            elif not freeze_llm_except_prefix or (use_prefix and freeze_llm_except_prefix):
                # Include LLM params if not freezing or if using prefix but not freezing except prefix
                llm_params.append(p)

        # Create parameter groups with different learning rates
        param_groups = []

        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.args.learning_rate,
                'weight_decay': self.args.weight_decay,
                'name': 'encoder'
            })

        if prefix_params:
            # Prefix parameters get higher learning rate initially for faster adaptation
            prefix_lr = getattr(self.args, 'prefix_lr', self.args.learning_rate * 2.0)
            param_groups.append({
                'params': prefix_params,
                'lr': prefix_lr,
                'weight_decay': getattr(self.args, 'prefix_weight_decay', self.args.weight_decay),
                'name': 'prefix'
            })

        if llm_params and not freeze_llm_except_prefix:
            # Lower learning rate for LLM parameters
            llm_lr = getattr(self.args, 'llm_lr', self.args.learning_rate * 0.1)
            param_groups.append({
                'params': llm_params,
                'lr': llm_lr,
                'weight_decay': getattr(self.args, 'llm_weight_decay', self.args.weight_decay * 0.1),
                'name': 'llm'
            })

        # Print parameter information
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print("Parameter groups:")
            for group in param_groups:
                print(f"  {group['name']}: {len(group['params'])} params, lr={group['lr']}")
                for p in group['params']:
                    print(f"    {p.shape}, dtype={p.dtype}")

        model_optim = optim.AdamW(param_groups)

        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('Optimizer created with parameter groups:', [g['name'] for g in param_groups])

        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MSE+DIFF':
            # Combined MSE with first-difference alignment
            diff_alpha = getattr(self.args, 'diff_loss_alpha', 0.1)
            return lambda pred, true: first_difference_alignment_loss(pred, true, diff_alpha)
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

    def run_ablation_study(self, setting, ablation_configs):
        """
        Run ablation studies for different prefix configurations.

        Args:
            setting: Base setting name
            ablation_configs: List of config dictionaries for different ablation experiments
        """
        results = {}

        for config in ablation_configs:
            print(f"\n=== Running ablation: {config['name']} ===")

            # Temporarily modify args for this ablation
            original_args = {}
            for key, value in config.items():
                if key != 'name' and hasattr(self.args, key):
                    original_args[key] = getattr(self.args, key)
                    setattr(self.args, key, value)

            # Rebuild model with new config
            self.model = self._build_model()

            # Train model
            trained_model = self.train(f"{setting}_{config['name']}")

            # Evaluate on different scenarios
            eval_results = self.evaluate_ablation_scenarios(trained_model, config['name'])
            results[config['name']] = eval_results

            # Restore original args
            for key, value in original_args.items():
                setattr(self.args, key, value)

        return results

    def evaluate_ablation_scenarios(self, model, config_name):
        """
        Evaluate model performance across different scenarios for ablation studies.
        """
        scenarios = {
            'hourly': {'data_path': 'ETTh1.csv', 'freq': 'h'},
            'daily': {'data_path': 'ETTm1.csv', 'freq': 'd'},
            'weekly': {'data_path': 'ETTh1.csv', 'freq': 'h'},  # Using hourly data but could be adapted
            'extreme_weather': {'data_path': 'weather.csv', 'freq': 'h'},  # Weather data for extreme conditions
        }

        results = {}

        for scenario_name, scenario_config in scenarios.items():
            try:
                # Set up data for this scenario
                original_data_path = self.args.test_data_path
                self.args.test_data_path = scenario_config['data_path']

                # Get test data
                self.args.root_path = './dataset/tsf'
                self.args.data_path = scenario_config['data_path']
                self.args.data = 'tsf'
                test_data, test_loader = self._get_data('test')

                # Evaluate
                preds, trues = self._evaluate_model(model, test_loader)

                # Calculate metrics
                mse = np.mean((preds - trues) ** 2)
                mae = np.mean(np.abs(preds - trues))
                smape_val = SMAPE(preds, trues)
                mape_val = MAPE(preds, trues)

                results[scenario_name] = {
                    'mse': mse,
                    'mae': mae,
                    'smape': smape_val,
                    'mape': mape_val
                }

                # Restore original data path
                self.args.test_data_path = original_data_path

            except Exception as e:
                print(f"Error evaluating {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}

        return results

    def _evaluate_model(self, model, test_loader):
        """Helper method to evaluate model and get predictions/trues"""
        preds = []
        trues = []

        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, None, None, None)
                else:
                    outputs = model(batch_x, None, None, None)

                outputs = outputs[:, -self.args.test_pred_len:, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        return preds, trues

    def get_default_ablation_configs(self):
        """
        Get default ablation study configurations.
        """
        configs = [
            {
                'name': 'no_prefix',
                'use_prefix': False,
                'loss': 'MSE'
            },
            {
                'name': 'prefix_shallow',
                'use_prefix': True,
                'prefix_injection_mode': 'shallow',
                'loss': 'MSE'
            },
            {
                'name': 'prefix_deep',
                'use_prefix': True,
                'prefix_injection_mode': 'deep',
                'loss': 'MSE'
            },
            {
                'name': 'prefix_with_diff_loss',
                'use_prefix': True,
                'prefix_injection_mode': 'deep',
                'loss': 'MSE+DIFF',
                'diff_loss_alpha': 0.1
            }
        ]
        return configs

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        self.args.root_path = './dataset/tsf'
        self.args.data_path = self.args.test_data_path
        self.args.data = 'tsf'
        test_data2, test_loader2 = self._get_data(flag='test')
        
        self.args.data = 'tsf_icl'
        test_data3, test_loader3 = self._get_data(flag="test")
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()

        # Enhanced scheduler with warmup for prefix training
        use_warmup = getattr(self.args, 'use_warmup', False)
        warmup_epochs = getattr(self.args, 'warmup_epochs', 5)

        if use_warmup:
            # Use warmup scheduler for stable prefix training
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                else:
                    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (self.args.train_epochs - warmup_epochs)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(model_optim, lr_lambda)
        else:
            scheduler = CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        criterion = self._select_criterion(self.args.loss)

        # Progressive unfreezing setup
        use_progressive_unfreezing = getattr(self.args, 'use_progressive_unfreezing', False)
        unfreeze_epochs = getattr(self.args, 'unfreeze_epochs', [30, 60])  # epochs to unfreeze more parameters
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) 

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

            # Progressive unfreezing
            if use_progressive_unfreezing and epoch in unfreeze_epochs:
                self._progressive_unfreeze(epoch, unfreeze_epochs)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, None, None, None)
            else:
                outputs = self.model(batch_x, None, None, None)

            # Handle different loss function signatures
            if callable(criterion) and 'first_difference' in str(criterion):
                # Custom loss function for prefix training
                loss = criterion(outputs, batch_y)
            else:
                # Standard loss function
                loss = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion) # test_loss indicates the result on the source datasets
            test_loss = vali_loss
            test_loss2 = self.vali2(test_data2, test_loader2, zero_shot_smape_loss())  # test_loss2 indicates the result on the target datasets
            test_loss3 = self.vali2(test_data3, test_loader3, zero_shot_smape_loss())  # test_loss3 indicates the result on the target datasets with time series prompts
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Zero Shot Test Loss: {4:.7f} In Context Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss2, test_loss3))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # Enhanced scheduler stepping
            if use_warmup or self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    # Print learning rates for all parameter groups
                    for i, group in enumerate(model_optim.param_groups):
                        lr = group['lr']
                        group_name = group.get('name', f'group_{i}')
                        print(f"{group_name} lr = {lr:.10f}")
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path), strict=False)

        return self.model

    def _progressive_unfreeze(self, current_epoch, unfreeze_epochs):
        """
        Progressively unfreeze more parameters during training for better adaptation.
        """
        if not hasattr(self.args, 'freeze_llm_except_prefix') or not self.args.freeze_llm_except_prefix:
            return

        # Determine which parameters to unfreeze based on epoch
        epoch_idx = unfreeze_epochs.index(current_epoch) if current_epoch in unfreeze_epochs else -1

        if epoch_idx >= 0:
            unfrozen_count = 0
            for name, param in self.model.named_parameters():
                if 'gpt2' in name and not param.requires_grad:
                    # Unfreeze some LLM parameters
                    if epoch_idx == 0:
                        # First unfreeze: only attention layers
                        if 'attn' in name:
                            param.requires_grad = True
                            unfrozen_count += 1
                    elif epoch_idx >= 1:
                        # Second unfreeze: all remaining layers
                        param.requires_grad = True
                        unfrozen_count += 1

            if unfrozen_count > 0 and ((self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu):
                print(f"Epoch {current_epoch}: Unfroze {unfrozen_count} additional parameters")

                # Update optimizer to include newly unfrozen parameters
                self.model_optim = self._select_optimizer()

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            
            outputs = torch.zeros((B, self.args.seq_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    for i in range(len(id_list) - 1):
                        outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None, None, None).detach().cpu()
            else:
                for i in range(len(id_list) - 1):
                    outputs[id_list[i]:id_list[i + 1], :, :] = self.model(x[id_list[i]:id_list[i + 1]], None, None, None).detach().cpu()
            pred = outputs[:, -self.args.token_len:, :]
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def vali2(self, vali_data, vali_loader, criterion):
        total_loss = []
        count= []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs[:, -self.args.test_pred_len:, :].detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                count.append(batch_x.shape[0])
        total_loss = np.average(total_loss, weights=count)
        self.model.train()
        
        return total_loss    

    def test_(self, test_loader):
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, None, None, None)
                else:
                    outputs = self.model(batch_x, None, None, None)

                outputs = outputs[:, -self.args.test_pred_len:, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        smape = SMAPE(preds, trues)
        mape = MAPE(preds, trues)
        print('mape:{:4f}, smape:{:.4f}'.format(mape, smape))
        
    def test(self, setting, test=0):
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, best_model_path)), strict=False)

        self.args.data_path = self.args.test_data_path
        
        self.args.root_path = './dataset/tsf'
        self.args.data_path = self.args.test_data_path
        self.args.data = 'tsf'
        test_data, test_loader = self._get_data('test')
        self.args.data = 'tsf_icl'
        test_data2, test_loader2 = self._get_data('test')
        
        print("zero shot forecasting")
        self.test_(test_loader)
        print("in context forecasting")
        self.test_(test_loader2)

