import os
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm

class CachedDataset(Dataset):
    def __init__(self, dataset, cache_path="cached_dataset.pt", force_reload=False):
        self.cache_path = cache_path

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
                sample_tensor, normal_tensor, class_tensor  = dataset[i]
                self.data.append((sample_tensor, normal_tensor, class_tensor))  # 리스트에 추가

            print("Cached Complete! Saving")
            torch.save(self.data, cache_path)  # `.pt` 파일로 저장
            print(f"Dataset Saved Complete : {cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]  # (tensor, meta_data) 형태 반환

class OrderFreqDataset(Dataset):
    def __init__(self, data_root, classes = ['normal', 'looseness', 'misalignment', 'unbalance', 'bearing'], averaging_size = 100, target_len=260, sensor_list=['motor_x', 'motor_y'], max_order = 10):
        
        self.classes        = classes
        self.averaging_size = averaging_size
        self.target_len     = target_len
        self.sensor_list    = sensor_list
        self.max_order      = max_order
        
        dataset_col = []
        class_col = []
        specific_col = []
        load_col = []
        severity_col = []
        file_path_col = []

        for dataset_name in os.listdir(data_root):
            dataset_dir = os.path.join(data_root, dataset_name)
            
            for class_name in os.listdir(dataset_dir):
                class_dir = os.path.join(dataset_dir, class_name)
                
                file_path_list = []
                for file_name in os.listdir(class_dir):
                    specific_class, _, _, severity, load_condition, _ = os.path.split(file_name)[-1].split('_')
                    
                    specific_col.append(specific_class)
                    severity_col.append(severity)
                    load_col.append(load_condition)
                    
                    file_path = os.path.join(class_dir, file_name)
                    file_path_list.append(file_path)
                    
                class_col += ([class_name]*len(file_path_list))
                dataset_col += ([dataset_name]*len(file_path_list))
                file_path_col += file_path_list


        self.dataset_df = pd.DataFrame({
            'dataset' : dataset_col,
            'class_name' : class_col,
            'specific_class' : specific_col,
            'load_condition' : load_col,
            'severity' : severity,
            'file_path' : file_path_col
        })
    
    def __len__(self):
        return len(self.dataset_df)
        
    def __getitem__(self, index):
        
        row = self.dataset_df.iloc()[index]
        
        sample_np, normal_np, data_info = self.sampling_smoothing(row)
        class_name = data_info['class_name']
        
        sample_tensor = torch.tensor(sample_np, dtype=torch.float32)
        normal_tensor = torch.tensor(normal_np, dtype=torch.float32)
        class_tensor = torch.tensor(self.classes.index(class_name), dtype=torch.long)
        
        return sample_tensor, normal_tensor, class_tensor
        
        
    def sampling_smoothing(self, row):
        
        dataset = row['dataset']
        class_name = row['class_name']
        file_path = row['file_path']
        specific_class = row['specific_class']
        severity = row['severity']
        load_condition = row['load_condition']
        
        sample_mag, interpolated_freq, data_info = self.open_file(file_path)

        intra_samples = self.dataset_df[(self.dataset_df['dataset'] == dataset) & \
                                        (self.dataset_df['class_name'] == class_name) & \
                                        (self.dataset_df['specific_class'] == specific_class) & \
                                        (self.dataset_df['load_condition'] == load_condition) & \
                                        (self.dataset_df['severity'] == severity)    ].sample(n=self.averaging_size)
        intra_mag_list = []
        for sample in intra_samples['file_path']:
            interpolated_mag, _, _ = self.open_file(sample)
            intra_mag_list.append(interpolated_mag)
        intra_mag_np = np.array(intra_mag_list)

        normal_samples = self.dataset_df[(self.dataset_df['dataset'] == dataset) & (self.dataset_df['class_name'] == 'normal')].sample(n=self.averaging_size)
        normal_mag_list = []
        for sample in normal_samples['file_path']:
            interpolated_mag, _, _ = self.open_file(sample)
            normal_mag_list.append(interpolated_mag)
        normal_mag_np = np.array(normal_mag_list)
        
        intra_mean_mag =intra_mag_np.mean(axis=0)
        normal_mean_mag = normal_mag_np.mean(axis=0)
        smoothed_mag = (sample_mag + intra_mean_mag)/2
        
        # ratio = (smoothed_mag - normal_mean_mag)/normal_mean_mag.max()
        normalized_mag = smoothed_mag/normal_mean_mag.max()
        normalized_normal_mag = normal_mean_mag/normal_mean_mag.max()
        
        return normalized_mag, normalized_normal_mag, data_info
    
    def open_file(self, file_path):
    
        # 0. File Open & Parsing
        class_name, sampling_rate, rpm, severity, load_condition, _ = os.path.split(file_path)[-1].split('_')
        sampling_rate = float(sampling_rate[:-3])*1000
        rpm = float(rpm)
        specific_class_name = class_name
        class_name = class_name.split('-')[0]
        
        data_info = {
            'class_name' : class_name,
            'specific_class_name' : specific_class_name,
            'sampling_rate' : sampling_rate,
            'severity' : severity,
            'load_condition' : load_condition
        }
        
        file_pd = pd.read_csv(file_path)
        data = []
        for sensor_name in self.sensor_list:
            data.append(file_pd[sensor_name])
        data_np = np.array(data)
        
        # 1. FFT
        fft_result = np.fft.rfft(data_np, axis=1)
        fft_mag = np.abs(fft_result)
        fft_freqs = np.fft.rfftfreq(data_np.shape[1], 1/sampling_rate)

        # 2. Order Freq & Slicing
        faundamental_freq = rpm/60
        order_freq = fft_freqs/faundamental_freq
        mask = (0.5 < order_freq) & (order_freq < (self.max_order+0.5))
        order_mag = fft_mag[:,mask]
        order_freq = order_freq[mask]

        # 3. Interpolation
        interpolated_freq = np.linspace(order_freq[0], order_freq[-1], self.target_len)
        interpolated_mag = []
        for ch_data in order_mag:
            freq = interp1d(order_freq, ch_data, kind='linear', fill_value='extrapolate')
            interpolated_ch = freq(interpolated_freq)
            interpolated_mag.append(interpolated_ch)
        interpolated_mag = np.array(interpolated_mag)

        return interpolated_mag, interpolated_freq, data_info
    
    
if __name__ == '__main__':
    
    dataset = OrderFreqDataset(
        data_root= '../dataset'
    )

    sample_tensor, normal_tensor, class_tensor = dataset[0]
    print(f'sample_tensor : {sample_tensor.shape}')
    print(f'normal_tensor : {normal_tensor.shape}')
    print(f'class : {class_tensor}')