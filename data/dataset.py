import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

import torch
import random
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import savgol_filter
from scipy.signal import stft
from scipy.signal import resample
import lightning as L

from collections import Counter

def count_classes(dataset):
    class_counter = Counter()
    for _, class_name, _ in dataset.indexed_file_list:
        class_counter[class_name] += 1
    return class_counter

class CachedDataset(Dataset):
    def __init__(self, dataset, cache_path="cached_dataset.pt", force_reload=False):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.data = []

        # 캐시된 파일이 존재하면 불러오기
        if not force_reload and os.path.exists(cache_path):
            print(f"Cached Dataset Loading... :{cache_path}")
            self.data = torch.load(cache_path, weights_only=False)
            print("Cached Dataset Load Complete!")
        else:
            print("Caching... (Dataset to Memory)")
            self.data = []

            # 데이터셋을 (tensor, dict) 형태로 저장
            for i in tqdm(range(len(dataset)), desc="Dataset Caching", unit="samples"):
                signal_tensor, signal_info, ref_tensor, ref_info  = dataset[i]
                self.data.append((signal_tensor, signal_info, ref_tensor, ref_info))  # 리스트에 추가

            print("Cached Complete! Saving")
            torch.save(self.data, cache_path)  # `.pt` 파일로 저장
            print(f"Dataset Saved Complete : {cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]  # (tensor, meta_data) 형태 반환

class VibrationDataset(Dataset):
    def __init__(self, data_root, 
                    dataset_used    = ['dxai', 'iis', 'mfd', 'vat', 'vbl'], 
                    class_used      = ['looseness', 'normal', 'unbalance','misalignment', 'bearing'], 
                    ch_used         = ['motor_x', 'motor_y'],
                    ref_class       = 'normal'):
        
        data_dict = {}
        
        for dataset_name in dataset_used:
            dataset_dir = os.path.join(data_root, dataset_name)
            dataset_dict = {}
            for class_name in class_used:
                class_dir = os.path.join(dataset_dir, class_name)
                data_cnt = 0
                if os.path.isdir(class_dir): # 클래스 폴더는 있을 수도 있고 없을 수도 있음
                    file_list = []
                    for file_name in os.listdir(class_dir):
                        file_path = os.path.join(class_dir, file_name)
                        file_list.append(file_path)
                        data_cnt+=1

                    dataset_dict[class_name] = file_list
            data_dict[dataset_name] = dataset_dict

        self.ch_used = ch_used
        self.indexed_file_list = []
        self.ref_dict = {}  # {dataset_name: [file_path, file_path, ...]} for ref_class

        for dataset_name, class_dict in data_dict.items():
            for class_name, file_list in class_dict.items():
                for file_path in file_list:
                    self.indexed_file_list.append((dataset_name, class_name, file_path))

            # pairing 대상 class 데이터 미리 저장
            if ref_class in class_dict:
                self.ref_dict[dataset_name] = class_dict[ref_class]
            else:
                self.ref_dict[dataset_name] = []
        self.transform = OrderFreqPipeline()
        
    def __len__(self):
        return len(self.indexed_file_list)

    def __getitem__(self, idx):
        dataset_name, _, file_path = self.indexed_file_list[idx]

        # pairing class에서 랜덤 선택
        ref_list = self.ref_dict.get(dataset_name, [])
        ref_file_path = None
        if ref_list:
            ref_file_path = random.choice(ref_list)

        signal_np, signal_info = self.open_file(file_path)
        signal_info['dataset_name'] = dataset_name
        ref_np, ref_info = self.open_file(ref_file_path)

        if self.transform is not None:
            signal_tensor, _ = self.transform(signal_np, signal_info['sampling_rate'], signal_info['rpm']/60)
            ref_tensor, _ = self.transform(ref_np, ref_info['sampling_rate'], ref_info['rpm']/60)
        else:
            signal_tensor = torch.tensor(signal_np)
            ref_tensor = torch.tensor(ref_np)
        return signal_tensor, signal_info, ref_tensor, ref_info
    
    def open_file(self, file_path):
        file_pd = pd.read_csv(file_path)
        file_np = file_pd[self.ch_used].transpose().to_numpy()
        
        file_name = os.path.basename(file_path)
        class_name, sampling_rate, rpm, severity, load_condition, data_idx = file_name.split('_')
        sampling_rate = float(sampling_rate[:-3])*1000
        rpm = float(rpm)
        class_name = class_name.split('-')[0]
        
        meta_info = {
            'class_name' : class_name,
            'sampling_rate' : sampling_rate,
            'rpm' : rpm,
            'severity' : severity,
            'load_condition' : load_condition,
            'data_idx' : data_idx
        }
        
        return file_np, meta_info
    
    
class OrderFreqPipeline:
    def __init__(self, harmonics=8, points_per_harmonic=32):
        self.harmonics = harmonics
        self.points_per_harmonic = points_per_harmonic
        
    def scale_to_detailed_harmonics(self, magnitude, freqs, sync_freq):
        """
        Scale FFT magnitude to detailed harmonics of sync_freq with interpolation.

        Args:
        - magnitude (np.ndarray): FFT magnitude of shape (channels, freq_bins).
        - freqs (np.ndarray): Corresponding frequency values.
        - sync_freq (float): Rotational synchronous frequency.

        Returns:
        - np.ndarray: Scaled FFT magnitude with detailed harmonic bins.
        - np.ndarray: Frequencies corresponding to the detailed harmonic bins.
        """
        # Define harmonic range starting at 0.5 * sync_freq
        detailed_bins = []
        for i in range(0, self.harmonics):
            harmonic_start = (i + 0.5) * sync_freq
            harmonic_end = (i + 1) * sync_freq
            detailed_bins.append(np.linspace(harmonic_start, harmonic_end, self.points_per_harmonic))

        # Flatten harmonic bins
        detailed_bins = np.concatenate(detailed_bins)

        # Interpolate magnitude to match detailed harmonic bins
        resampled_magnitude = np.array([
            np.interp(detailed_bins, freqs, magnitude[channel])
            for channel in range(magnitude.shape[0])
        ])
        return resampled_magnitude, detailed_bins

    def __call__(self, data_np, sampling_rate, sync_freq):
        """
        Process a single tensor of vibration data.

        Args:
        - data (torch.Tensor): Input data of shape (channels, time_steps).
        - sampling_rate (float): Sampling rate of the data in Hz.
        - sync_freq (float): Rotational synchronous frequency in Hz.

        Returns:
        - torch.Tensor: Processed data of shape (channels, harmonics * points_per_harmonic).
        - np.ndarray: Frequencies corresponding to the processed data.
        """

        # Step 1: FFT
        fft_result = np.fft.rfft(data_np, axis=1)
        magnitude = np.abs(fft_result)

        # Step 2: Frequency bins
        freqs = np.fft.rfftfreq(data_np.shape[1], 1 / sampling_rate)
        
        # Step 3: Scale to detailed harmonics
        detailed_magnitude, detailed_bins = self.scale_to_detailed_harmonics(magnitude, freqs, sync_freq)

        
        return torch.tensor(detailed_magnitude, dtype=torch.float32), detailed_bins

class StatisticPipeline:
    def __init__(self, data_type='freq'):
        self.data_type = data_type
        
    def extract_frequency_domain_features_multichannel(self, signals, sampling_rate=4096):
        """
        2채널 주파수 도메인 특징 추출
        :param signals: numpy array (2 x N)
        :return: (Tensor shape: [2, feature_dim], List of feature dicts per channel)
        """

        feature_list = []
        feature_tensors = []

        for signal in signals:
            N = len(signal)
            freq = np.fft.fftfreq(N, d=1/sampling_rate)[:N//2]
            fft_values = np.abs(fft(signal))[:N//2]

            total_power = np.sum(fft_values**2)
            max_frequency = freq[np.argmax(fft_values)]
            mean_frequency = np.sum(freq * fft_values) / np.sum(fft_values)
            median_frequency = freq[np.cumsum(fft_values) >= np.sum(fft_values) / 2][0]
            spectral_skewness = np.mean((freq - mean_frequency)**3 * fft_values) / (np.std(freq) + 1e-10)**3
            spectral_kurtosis = np.mean((freq - mean_frequency)**4 * fft_values) / (np.std(freq) + 1e-10)**4
            peak_amplitude = np.max(fft_values)
            band_energy = np.sum(fft_values[(freq >= 0.1) & (freq <= 1.0)]**2)
            dominant_frequency_power = fft_values[np.argmax(fft_values)]**2
            spectral_entropy = -np.sum((fft_values / np.sum(fft_values)) * np.log2(fft_values / np.sum(fft_values) + 1e-10))
            rms_frequency = np.sqrt(np.mean(fft_values**2))
            variance_frequency = np.var(fft_values)

            features = {
                "total_power": total_power,
                "max_frequency": max_frequency,
                "mean_frequency": mean_frequency,
                "median_frequency": median_frequency,
                "spectral_skewness": spectral_skewness,
                "spectral_kurtosis": spectral_kurtosis,
                "peak_amplitude": peak_amplitude,
                "band_energy_0.1_1Hz": band_energy,
                "dominant_frequency_power": dominant_frequency_power,
                "spectral_entropy": spectral_entropy,
                "rms_frequency": rms_frequency,
                "variance_frequency": variance_frequency
            }

            feature_tensors.append(torch.tensor(list(features.values()), dtype=torch.float32))
            feature_list.append(features)

        return torch.stack(feature_tensors), feature_list

    def extract_time_domain_features_multichannel(self, signals):
        """
        2채널 시간 도메인 특징 추출
        :param signals: numpy array (2 x N)
        :return: (Tensor shape: [2, feature_dim], List of feature dicts per channel)
        """
        feature_list = []
        feature_tensors = []

        for signal in signals:
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            signal_max = np.max(signal)
            signal_min = np.min(signal)
            signal_rms = np.sqrt(np.mean(signal**2))
            signal_skew = np.mean((signal - signal_mean)**3) / (signal_std**3 + 1e-10)
            signal_kurt = np.mean((signal - signal_mean)**4) / (signal_std**4 + 1e-10)
            signal_peak = np.max(np.abs(signal))
            signal_ppv = signal_max - signal_min
            signal_crest = signal_peak / (signal_rms + 1e-10)
            signal_impulse = signal_peak / (np.mean(np.abs(signal)) + 1e-10)
            signal_shape = signal_rms / (np.mean(np.abs(signal)) + 1e-10)

            features = {
                "mean": signal_mean,
                "std": signal_std,
                "max": signal_max,
                "min": signal_min,
                "rms": signal_rms,
                "skewness": signal_skew,
                "kurtosis": signal_kurt,
                "peak": signal_peak,
                "ppv": signal_ppv,
                "crest_factor": signal_crest,
                "impulse_factor": signal_impulse,
                "shape_factor": signal_shape
            }

            feature_tensors.append(torch.tensor(list(features.values()), dtype=torch.float32))
            feature_list.append(features)

        return torch.stack(feature_tensors), feature_list

    def compute_fft_multichannel(signals: np.ndarray, sampling_rate: float):
        """
        다채널 FFT 변환 수행.
        
        :param signals: np.ndarray, shape (2, N)
        :param sampling_rate: float, 샘플링 레이트 (Hz)
        :return: list of (magnitude, phase, freq) tuples
        """
        results = []

        for signal in signals:
            n = len(signal)
            fft_result = np.fft.fft(signal)
            freq = np.fft.fftfreq(n, d=1 / sampling_rate)
            magnitude = np.abs(fft_result)
            phase = np.angle(fft_result)

            half_n = n // 2
            results.append((magnitude[:half_n], phase[:half_n], freq[:half_n]))

        return results

    def __call__(self, data_np, sampling_rate, sync_freq):
        """
        Process a single tensor of vibration data.

        Args:
        - data (torch.Tensor): Input data of shape (channels, time_steps).
        - sampling_rate (float): Sampling rate of the data in Hz.
        - sync_freq (float): Rotational synchronous frequency in Hz.

        Returns:
        - torch.Tensor: Processed data of shape (channels, harmonics * points_per_harmonic).
        - np.ndarray: Frequencies corresponding to the processed data.
        """
        
        if self.data_type == 'freq':
            feature_tensor, feature_dict = self.extract_frequency_domain_features_multichannel(signals=data_np, sampling_rate=sampling_rate)
        elif self.data_type == 'time':
            feature_tensor, feature_dict = self.extract_time_domain_features_multichannel(signals=data_np)
        else:
            print(f'Worng feature domain : {self.data_type}')
        
        return feature_tensor, feature_dict



if __name__=='__main__':
    dataset_root = '/home/data/'
    preprocessing_type = 'order_freq'
    data_type = 'freq' 

    if preprocessing_type == 'order_freq':
        pipeline = OrderFreqPipeline(harmonics=8, points_per_harmonic=32)
    elif preprocessing_type == 'statistic':
        pipeline = StatisticPipeline(data_type=data_type)
    else:
        print(f'Wrong preprocessing type : {preprocessing_type}')

    
    train_dataset = VibrationDataset(dataset_root, 
                                    dataset_used = ['dxai', 'iis', 'mfd', 'vat', 'vbl'], 
                                    ch_used=['motor_x', 'motor_y'], 
                                    class_used=['looseness', 'normal', 'unbalance', 'misalignment', 'bearing']
                                    )
    for batch in tqdm(train_dataset,desc="for checking Nan"):
        signal_tensor, signal_info, ref_tensor, ref_info = batch
        if torch.isnan(signal_tensor).any():
            print("NaN detected in signal_tensor. Breaking.")
            break
    print(signal_tensor.shape)