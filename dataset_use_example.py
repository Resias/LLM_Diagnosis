import os
import ast

import pandas as pd
import torch

from data.dataset import OrderInvariantSignalImager, WindowedVibrationDataset, visualize_imaging_tensor

if __name__ == '__main__':
    data_root = os.path.join(os.getcwd(), 'data', 'processed')
    meta_csv = os.path.join(data_root, 'meta.csv')
    meta_pd = pd.read_csv(meta_csv)
    data_mode = 'stft+cross'

    meta_pd['sensor_position'] = meta_pd['sensor_position'].apply(ast.literal_eval)
    meta_pd = meta_pd[5 <= meta_pd['data_sec']]

    # 필요한 데이터셋만 남기고 나머지는 제거
    using_dataset = ['dxai', 'iis', 'vat', 'vbl', 'mfd']
    meta_pd = meta_pd[meta_pd['dataset'].isin(using_dataset)]
    meta_pd['class_name'].unique()
    
    signal_imger = OrderInvariantSignalImager(
        mode=data_mode,
        log1p=True,
        normalize="per_channel",  # None | "per_channel" | "global"
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=20.0,           # order 축 상한
        H_out=128,                # order-bin 수
        W_out=128,                # time-bin 수
        # STFT
        stft_nperseg=256,
        stft_hop=128,
        stft_window="hann",
        stft_center=True,
        stft_power=1.0,           # 1: magnitude, 2: power
        # CWT
        cwt_wavelet="morl",
        cwt_num_scales=64,
        cwt_scale_base=2.0,
    )

    dataset = WindowedVibrationDataset(
        meta_df=meta_pd,
        data_root=data_root,
        window_sec=5,
        stride_sec=2,
        cache_mode='file',
        transform=signal_imger
    )

    # 단일 샘플 미리보기
    x_tensor, y_tensor = dataset[100]  # (C,H,W)
    print(y_tensor)
    visualize_imaging_tensor(x_tensor, mode=data_mode, max_order=20, window_sec=5.0, save_path='test')
    print(x_tensor.shape)
    print(meta_pd['class_name'])