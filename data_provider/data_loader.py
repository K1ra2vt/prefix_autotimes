from multiprocessing import context
import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
from models.prefix import extract_structured_context_features
import warnings
import calendar
from transformers import GPT2Tokenizer

warnings.filterwarnings('ignore')

# 抽取时间序列数据的一些统计上的特征，这部分代码目前是cursor写的，没有经过检查
def extract_context_features(df_raw, data_name, start_idx, end_idx, seq_len):
    """
    Extract context features for prefix generation from the dataset.
    Returns a dictionary of normalized context features for the time window.
    """
    context_features = {}

    # Time-based features
    if 'date' in df_raw.columns:
        dates = pd.to_datetime(df_raw['date'].iloc[start_idx:end_idx])

        # Basic time features
        context_features['hour'] = dates.dt.hour.values / 23.0  # normalize to [0,1]
        context_features['day_of_week'] = dates.dt.dayofweek.values / 6.0  # normalize to [0,1]
        context_features['month'] = dates.dt.month.values / 12.0  # normalize to [0,1]
        context_features['day_of_month'] = dates.dt.day.values / 31.0  # normalize to [0,1]

        # Holiday detection (simplified - weekends and major holidays)
        is_weekend = dates.dt.dayofweek.isin([5, 6]).astype(int)
        # Simple holiday detection for common holidays (can be extended)
        is_holiday = is_weekend | ((dates.dt.month == 1) & (dates.dt.day == 1)) | \
                    ((dates.dt.month == 12) & (dates.dt.day == 25)) | \
                    ((dates.dt.month == 7) & (dates.dt.day == 4))
        context_features['is_holiday'] = is_holiday.values.astype(float)

    # Dataset-specific features
    if data_name.lower().startswith('weather'):
        # Meteorological features from weather.csv
        weather_cols = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
                       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
                       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
                       'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m²)',
                       'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)', 'Tlog (degC)']

        for col in weather_cols:
            if col in df_raw.columns:
                values = df_raw[col].iloc[start_idx:end_idx].values
                # Normalize using robust statistics
                median_val = np.median(values)
                mad_val = np.median(np.abs(values - median_val))
                if mad_val > 0:
                    context_features[f'weather_{col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("²", "2").replace("µ", "u")}'] = (values - median_val) / mad_val
                else:
                    context_features[f'weather_{col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("²", "2").replace("µ", "u")}'] = values - median_val

        # Statistical aggregations over the sequence window
        if len(weather_cols) > 0:
            available_cols = [col for col in weather_cols if col in df_raw.columns]
            if available_cols:
                seq_data = df_raw[available_cols].iloc[start_idx:end_idx].values
                context_features['weather_mean_temp'] = np.mean(seq_data[:, [col in ['T (degC)', 'Tpot (K)', 'Tdew (degC)', 'Tlog (degC)'] for col in available_cols]], axis=1) / 50.0
                context_features['weather_mean_humidity'] = np.mean(seq_data[:, [col in ['rh (%)', 'VPact (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)'] for col in available_cols]], axis=1) / 100.0
                context_features['weather_max_wind'] = np.max(seq_data[:, [col in ['wv (m/s)', 'max. wv (m/s)'] for col in available_cols]], axis=1) / 50.0
                context_features['weather_total_rain'] = np.sum(seq_data[:, [col in ['rain (mm)'] for col in available_cols]], axis=1) / 100.0

    elif data_name.lower().startswith('ett'):
        # ETT dataset features (energy load features)
        ett_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        for col in ett_cols:
            if col in df_raw.columns:
                values = df_raw[col].iloc[start_idx:end_idx].values
                # Normalize using min-max scaling per sequence
                min_val, max_val = np.min(values), np.max(values)
                if max_val > min_val:
                    context_features[f'ett_{col.lower()}'] = (values - min_val) / (max_val - min_val)
                else:
                    context_features[f'ett_{col.lower()}'] = values - min_val

        # Statistical aggregations
        if ett_cols and any(col in df_raw.columns for col in ett_cols):
            available_cols = [col for col in ett_cols if col in df_raw.columns]
            seq_data = df_raw[available_cols].iloc[start_idx:end_idx].values
            context_features['ett_load_mean'] = np.mean(seq_data, axis=1)
            context_features['ett_load_std'] = np.std(seq_data, axis=1)
            context_features['ett_load_trend'] = np.polyfit(np.arange(len(seq_data)), np.mean(seq_data, axis=1), 1)[0] if len(seq_data) > 1 else 0.0

    elif data_name.lower().startswith('electricity'):
        # Electricity consumption features (multi-variate)
        electricity_cols = [str(i) for i in range(320)]  # 0-319 columns
        available_cols = [col for col in electricity_cols if col in df_raw.columns]

        if available_cols:
            seq_data = df_raw[available_cols].iloc[start_idx:end_idx].values
            # Aggregate statistics across all electricity channels
            context_features['electricity_mean'] = np.mean(seq_data, axis=1)
            context_features['electricity_std'] = np.std(seq_data, axis=1)
            context_features['electricity_max'] = np.max(seq_data, axis=1)
            context_features['electricity_min'] = np.min(seq_data, axis=1)
            # Peak hour detection (simplified)
            if 'hour' in context_features:
                peak_hours = ((context_features['hour'] * 23 >= 17) & (context_features['hour'] * 23 <= 21)).astype(float)
                context_features['electricity_is_peak_hour'] = peak_hours

    # Convert all features to torch tensors and stack them
    # Convert to tensor format expected by the model
    # --- 这里删掉
    # feature_values = []
    # for key, values in context_features.items():
    #     if isinstance(values, np.ndarray) and len(values) > 0:
    #         # Take mean of the feature values to get a scalar per feature
    #         feature_values.append(float(np.mean(values)))
    #     elif hasattr(values, '__len__') and len(values) > 0:
    #         feature_values.append(float(np.mean(values)))
    #     elif isinstance(values, (int, float)):
    #         feature_values.append(float(values))

    # if feature_values:
    #     # Return tensor of feature means
    #     return torch.tensor(feature_values, dtype=torch.float32)
    # else:
    #     # Return zero tensor if no features
    #     return torch.zeros(4, dtype=torch.float32)  # fallback to basic stats size
    return context_features


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False, model=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.model = model

        # Initialize tokenizer for structured prompts if using prefix
        self.tokenizer = None
        if model == 'AutoTimes_Gpt2':
            # Will be initialized later when needed
            pass

        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1


    def __read_data__(self):
        self.scaler = StandardScaler()
        self.df_raw = pd.read_csv(os.path.join(self.root_path,
                                               self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        self.border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = self.df_raw.columns[1:]
        df_data = self.df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0]
        # 根据模型类型加载对应的预处理文件
        stamp_file = f'{data_name}.pt'
        if self.model == 'AutoTimes_Gpt2':
            stamp_file = f'{data_name}.pt'
        elif self.model == 'AutoTimes_Llama':
            stamp_file = f'{data_name}.pt'
        self.data_stamp = torch.load(os.path.join(self.root_path, stamp_file))
        self.data_x = data[self.border1:border2]
        self.data_y = data[self.border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        # Extract structured context features for prefix generation
        if self.model == 'AutoTimes_Gpt2':
            # Initialize tokenizer if not already done
            if self.tokenizer is None:
                try:
                    # Try to load tokenizer from the same path as the model
                    tokenizer_path = '/home/u4_3090_4/baseModel_gpt2'  # Default path
                    self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                except Exception as e:
                    print(f"Warning: Could not load tokenizer: {e}")
                    self.tokenizer = None

            # Extract time series data for this sample
            time_series_sample = torch.from_numpy(seq_x).unsqueeze(0)  # [1, seq_len, 1]
            dataset_name = self.data_path.split('.')[0]

            # Get tokenized structured prompts
            context_dict = extract_structured_context_features(
                time_series_sample, dataset_name, self.seq_len, self.pred_len, self.tokenizer
            )

            # Return tokens and attention masks for prefix generation
            if self.tokenizer is not None and 'tokens' in context_dict:
                context_features = {
                    'tokens': context_dict['tokens'],  # [max_length]
                    'attention_mask': context_dict['attention_masks'],  # [max_length]
                    'max_length': context_dict['max_length']
                }
            else:
                # Fallback to original numeric features
                context_features = extract_context_features(
                    self.df_raw, dataset_name,
                    self.border1 + s_begin, self.border1 + s_end, self.seq_len
                )
        else:
            # For non-GPT2 models, use original context features
            context_features = extract_context_features(
                self.df_raw, self.data_path.split('.')[0],
                self.border1 + s_begin, self.border1 + s_end, self.seq_len
            )

        return seq_x, seq_y, seq_x_mark, seq_y_mark, context_features

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False, model=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.model = model
        self.root_path = root_path
        self.data_path = data_path

        # Initialize tokenizer for structured prompts if using prefix
        self.tokenizer = None
        if model == 'AutoTimes_Gpt2':
            # Will be initialized later when needed
            pass

        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1


    def __read_data__(self):
        self.scaler = StandardScaler()
        self.df_raw = pd.read_csv(os.path.join(self.root_path,
                                               self.data_path))
        num_train = int(len(self.df_raw) * 0.7)
        num_test = int(len(self.df_raw) * 0.2)
        num_vali = len(self.df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(self.df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(self.df_raw)]
        self.border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        cols_data = self.df_raw.columns[1:]
        df_data = self.df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0]
        if self.model == 'AutoTimes_Gpt2':
            stamp_file = f'{data_name}.pt'
        self.data_stamp = torch.load(os.path.join(self.root_path, stamp_file))
        self.data_stamp = self.data_stamp[self.border1:border2]
        self.data_x = data[self.border1:border2]
        self.data_y = data[self.border1:border2]


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        # Extract structured context features for prefix generation
        if self.model == 'AutoTimes_Gpt2':
            # Initialize tokenizer if not already done
            if self.tokenizer is None:
                try:
                    # Try to load tokenizer from the same path as the model
                    tokenizer_path = '/home/u4_3090_4/baseModel_gpt2'  # Default path
                    self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                except Exception as e:
                    print(f"Warning: Could not load tokenizer: {e}")
                    self.tokenizer = None

            # Extract time series data for this sample
            time_series_sample = torch.from_numpy(seq_x).unsqueeze(0)  # [1, seq_len, 1]
            dataset_name = self.data_path.split('.')[0]

            # Get tokenized structured prompts
            context_dict = extract_structured_context_features(
                time_series_sample, dataset_name, self.seq_len, self.pred_len, self.tokenizer
            )

            # Return tokens and attention masks for prefix generation
            if self.tokenizer is not None and 'tokens' in context_dict:
                context_features = {
                    'tokens': context_dict['tokens'],  # [max_length]
                    'attention_mask': context_dict['attention_masks'],  # [max_length]
                    'max_length': context_dict['max_length']
                }
            else:
                # Fallback to original numeric features
                context_features = extract_context_features(
                    self.df_raw, dataset_name,
                    self.border1 + s_begin, self.border1 + s_end, self.seq_len
                )
        else:
            # For non-GPT2 models, use original context features
            context_features = extract_context_features(
                self.df_raw, self.data_path.split('.')[0],
                self.border1 + s_begin, self.border1 + s_end, self.seq_len
            )

        return seq_x, seq_y, seq_x_mark, seq_y_mark, context_features

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seasonal_patterns=None, scale=True, drop_short=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv',
                 scale=False, inverse=False, seasonal_patterns='Yearly', drop_short=False, model=None):
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.model = model
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False, model=None):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        print(self.seq_len, self.label_len, self.pred_len)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self.__read_data__()
        self.model = model


    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        if self.drop_short:
            timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)

        return data_x, data_y, data_x, data_y

    def __len__(self):
        return self.tot_len

class Dataset_TSF_ICL(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=True, model=None):
        
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        return timeseries

    # we uniformly adopting the first time points of the time series as the corresponding prompt.
    def __getitem__(self, index):        
        data_x1 = self.timeseries[index][:2*self.token_len]
        data_x2 = self.timeseries[index][-2*self.token_len:-1*self.token_len]
        data_x = np.concatenate((data_x1, data_x2))
        data_y = self.timeseries[index][-1*self.token_len:]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        return data_x, data_y, data_x, data_y

    def __len__(self):
        return len(self.timeseries)

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        self.df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = self.df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        if self.data_set_type in ['traffic', 'electricity', 'ETTh1', 'ETTh2']:
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type == 'weather':
            end = (start + datetime.timedelta(minutes=10*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type in ['ETTm1', 'ETTm2']:
            end = (start + datetime.timedelta(minutes=15*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"

        # Extract context features for this time window (aggregated)
        context_features = extract_context_features(
            self.df_raw, self.data_set_type, s_begin, s_end, self.token_len
        )

        # Aggregate context features to single values per feature
        aggregated_features = {}
        for key, values in context_features.items():
            if isinstance(values, torch.Tensor):
                # Take mean across the time window for each feature
                aggregated_features[key] = torch.mean(values, dim=0, keepdim=True)
            else:
                aggregated_features[key] = torch.tensor(np.mean(values), dtype=torch.float32)

        return seq_x_mark, aggregated_features

    def __len__(self):
        return len(self.data_stamp)
