import pandas as pd
from tqdm import tqdm
import ast
import ast
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
import pywt

def _starts_for(n, win_n, stride_n, drop_last=True):
    if win_n <= 0 or stride_n <= 0: raise ValueError("win_n/stride_n must be > 0")
    if n < win_n: return []
    starts = list(range(0, n - win_n + 1, stride_n))
    if not drop_last and (n - win_n) % stride_n != 0:
        last_start = n - win_n
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
    return starts

def rolling_windows_1d(a, win_n, stride_n):
    # 반환: (num_windows, win_n) - zero-copy view (as_strided)
    n = a.shape[-1]
    num = (n - win_n) // stride_n + 1
    if num <= 0: return np.empty((0, win_n), dtype=a.dtype)
    shape = (num, win_n)
    strides = (a.strides[-1] * stride_n, a.strides[-1])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

class WindowedVibrationDataset(Dataset):
    def __init__(
        self, meta_df, data_root, window_sec, stride_sec,
        drop_last=True, dtype=torch.float32, transform=None,
        channel_order=("x","y"), cache_mode="file"  # 'none' | 'file' | 'windows'
    ):
        self.meta_df = meta_df.reset_index(drop=True).copy()
        
        
        # 클래스 통합 매핑
        self.class_list = ['normal', 'unbalance', 'looseness', 'misalignment', 'bearing']
        merge_map = {
            # 정상
            'normal': 'normal',

            # 언밸런스
            'unbalance': 'unbalance',
            'unbalalnce': 'unbalance',  # 오타 수정
            'imbalance': 'unbalance',

            # 루즈니스
            'looseness': 'looseness',

            # 미스얼라인먼트
            'misalignment': 'misalignment',
            'horizontal-misalignment': 'misalignment',
            'vertical-misalignment': 'misalignment',

            # 베어링
            'bpfo': 'bearing',
            'bpfi': 'bearing',
            'bearing': 'bearing',
            'overhang_cage_fault': 'bearing',
            'overhang_ball_fault': 'bearing',
            'overhang_outer_race': 'bearing',
            'underhang_cage_fault': 'bearing',
            'underhang_ball_fault': 'bearing',
            'underhang_outer_race': 'bearing',
        }

        # 새로운 컬럼 생성 (통합 클래스명)
        self.meta_df['merged_class'] = self.meta_df['class_name'].map(merge_map)
        
        
        if isinstance(self.meta_df.loc[0, "sensor_position"], str):
            self.meta_df["sensor_position"] = self.meta_df["sensor_position"].apply(ast.literal_eval)

        self.data_root   = data_root
        self.window_sec  = float(window_sec)
        self.stride_sec  = float(stride_sec)
        self.drop_last   = drop_last
        self.dtype       = dtype
        self.transform   = transform
        self.channel_order = channel_order
        self.cache_mode  = cache_mode

        self.index_map = []      # (row_idx, start)
        self._row_meta = {}      # per-row meta
        self._file_cache = {}    # cache_mode in ('file','windows'): row_idx -> np.ndarray or dict
        self._win_cache  = {}    # cache_mode == 'windows': row_idx -> (W, 2, win_n)

        # 인덱스/캐시 준비
        print('caching dataset ... ')
        for row_idx, row in tqdm(self.meta_df.iterrows()):
            file_path = os.path.join(self.data_root, row["file_name"])
            sr = float(row["sampling_rate"])
            sensor_pos = row["sensor_position"]
            dataset = row["dataset"]

            if dataset == "iis":
                x_idx = sensor_pos.index("disk_x"); y_idx = sensor_pos.index("disk_y")
            else:
                x_idx = sensor_pos.index("motor_x"); y_idx = sensor_pos.index("motor_y")

            arr = np.load(file_path, mmap_mode=("r" if self.cache_mode=="none" else None))
            n_samples = arr.shape[1]
            win_n   = int(round(self.window_sec * sr))
            stride_n= int(round(self.stride_sec * sr))
            starts  = _starts_for(n_samples, win_n, stride_n, self.drop_last)

            self._row_meta[row_idx] = {
                "file_path": file_path, "sr": sr,
                "x_idx": x_idx, "y_idx": y_idx, "win_n": win_n
            }
            for s in starts:
                self.index_map.append((row_idx, s))

            if self.cache_mode == "file":
                # 파일을 RAM로딩 (한 번만)
                if row_idx not in self._file_cache:
                    self._file_cache[row_idx] = np.load(file_path)  # (S, N), ndarray in RAM
            elif self.cache_mode == "windows":
                # 윈도우까지 프리컴퓨트 (최고속도/메모리 多)
                base = np.load(file_path)  # (S, N) in RAM
                x = base[x_idx]; y = base[y_idx]
                xw = rolling_windows_1d(x, win_n, stride_n)  # (W, win_n) - view
                yw = rolling_windows_1d(y, win_n, stride_n)
                # 채널 스택 (W, 2, win_n)
                if channel_order == ("x","y"):
                    win = np.stack([xw, yw], axis=1)
                else:
                    win = np.stack([yw, xw], axis=1)
                # as_strided view라 학습 중 쓰기 방지 위해 copy 권장
                self._win_cache[row_idx] = win.copy()

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx, return_info=False):
        row_idx, start = self.index_map[idx]
        row = self.meta_df.iloc[row_idx]
        meta = self._row_meta[row_idx]
        sr, x_idx, y_idx, win_n = meta["sr"], meta["x_idx"], meta["y_idx"], meta["win_n"]

        if self.cache_mode == "none":
            arr = np.load(meta["file_path"], mmap_mode="r")
            x_seg = arr[x_idx, start:start+win_n]
            y_seg = arr[y_idx, start:start+win_n]
            seg = np.stack([x_seg, y_seg], axis=0) if self.channel_order==("x","y") \
                  else np.stack([y_seg, x_seg], axis=0)

        elif self.cache_mode == "file":
            base = self._file_cache[row_idx]                   # (S, N) in RAM
            x_seg = base[x_idx, start:start+win_n]
            y_seg = base[y_idx, start:start+win_n]
            seg = np.stack([x_seg, y_seg], axis=0) if self.channel_order==("x","y") \
                  else np.stack([y_seg, x_seg], axis=0)

        else:  # 'windows'
            win = self._win_cache[row_idx]                     # (W, 2, win_n)
            # start는 샘플 인덱스, 윈도우 인덱스로 변환
            stride_n = int(round(self.stride_sec * sr))
            widx = (start // stride_n)
            seg = win[widx]  # (2, win_n)

        info = {
            "sampling_rate": sr,
            "rpm": float(row["rpm"]),
            "label_class": row["class_name"],
            "severity": row["severity"],
            "load_condition": row["load_condition"],
            "dataset": row["dataset"]
        }
        
        tensor_img = self.transform(seg, sr=sr, rpm=float(row["rpm"]))
        class_idx = self.class_list.index(row['merged_class'])
        tensor_cls = torch.tensor(class_idx ,dtype=torch.long)
        if return_info:
            return tensor_img, tensor_cls, info
        else:
            return tensor_img, tensor_cls

class OrderInvariantSignalImager:
    """
    seg: np.ndarray (2, T)  # [x, y]
    __call__(seg, sr, rpm) -> torch.Tensor (C, H_out, W_out)

    mode ∈ {
      "stft", "stft+cross", "stft_complex",
      "cwt",  "cwt+cross",  "cwt_complex",
    }

    파이프라인:
      1) STFT 또는 CWT 계수 계산 (+ 선택적 cross/complex 채널 구성)
      2) 주파수 배열(f) 또는 CWT 주파수(f_cwt)를 회전주파수 f_rot = rpm/60 으로 나눠 order 축으로 정규화
      3) order <= max_order 로 마스킹
      4) (order, time) 2D 리샘플로 (H_out, W_out) 고정
      5) log/정규화 후 텐서 반환
    """
    def __init__(
        self,
        mode="stft+cross",
        # 공통
        log1p=True,
        normalize="per_channel",  # None | "per_channel" | "global"
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=20.0,           # order 축 상한
        H_out=128,                # order-bin 수
        W_out=256,                # time-bin 수
        # STFT
        stft_nperseg=1024,
        stft_hop=256,
        stft_window="hann",
        stft_center=True,
        stft_power=1.0,           # 1: magnitude, 2: power
        # CWT
        cwt_wavelet="morl",
        cwt_num_scales=64,
        cwt_scale_base=2.0,
    ):
        assert mode in {
            "stft", "stft+cross", "stft_complex",
            "cwt",  "cwt+cross",  "cwt_complex",
        }
        self.mode = mode
        self.log1p = log1p
        self.normalize = normalize
        self.eps = eps
        self.out_dtype = out_dtype
        self.max_order = float(max_order)
        self.H_out = int(H_out)
        self.W_out = int(W_out)

        self.stft_nperseg = stft_nperseg
        self.stft_hop = stft_hop
        self.stft_window = stft_window
        self.stft_center = stft_center
        self.stft_power = stft_power

        self.cwt_wavelet = cwt_wavelet
        self.cwt_num_scales = cwt_num_scales
        self.cwt_scale_base = cwt_scale_base

    # ---------- 유틸 ----------
    def _apply_log_norm(self, x: np.ndarray) -> np.ndarray:
        # x: (C, H, W)
        if self.log1p:
            x = np.sign(x) * np.log1p(np.abs(x))
        if self.normalize == "per_channel":
            m = x.mean(axis=(1,2), keepdims=True)
            s = x.std(axis=(1,2), keepdims=True) + self.eps
            x = (x - m) / s
        elif self.normalize == "global":
            x = (x - x.mean()) / (x.std() + self.eps)
        return x.astype(np.float32)

    def _resize_CHW(self, arr: np.ndarray, H_new: int, W_new: int) -> np.ndarray:
        """
        arr: (C, H, W) → (C, H_new, W_new)  (2-pass 1D linear interp: 시간축→order축)
        """
        C, H, W = arr.shape
        # 1) 시간축 보간
        if W != W_new:
            x_old = np.linspace(0.0, 1.0, W, endpoint=True)
            x_new = np.linspace(0.0, 1.0, W_new, endpoint=True)
            out_t = np.empty((C, H, W_new), dtype=arr.dtype)
            for c in range(C):
                for h in range(H):
                    out_t[c, h] = np.interp(x_new, x_old, arr[c, h])
        else:
            out_t = arr

        # 2) order축 보간
        if H != H_new:
            y_old = np.linspace(0.0, 1.0, H, endpoint=True)
            y_new = np.linspace(0.0, 1.0, H_new, endpoint=True)
            out = np.empty((C, H_new, W_new), dtype=arr.dtype)
            for c in range(C):
                for w in range(W_new):
                    out[c, :, w] = np.interp(y_new, y_old, out_t[c, :, w])
        else:
            out = out_t
        return out

    # ---------- STFT ----------
    def _stft_xy(self, x: np.ndarray, y: np.ndarray, sr: float):
        pad = (self.stft_nperseg // 2) if self.stft_center else 0
        if pad > 0:
            x = np.pad(x, (pad, pad), mode="reflect")
            y = np.pad(y, (pad, pad), mode="reflect")
        win = get_window(self.stft_window, self.stft_nperseg, fftbins=True)
        noverlap = self.stft_nperseg - self.stft_hop
        f, t, X = stft(x, fs=sr, window=win, nperseg=self.stft_nperseg,
                       noverlap=noverlap, nfft=None, padded=False, boundary=None)
        _, _, Y = stft(y, fs=sr, window=win, nperseg=self.stft_nperseg,
                       noverlap=noverlap, nfft=None, padded=False, boundary=None)
        return f, t, X, Y

    def _build_stft_maps(self, seg: np.ndarray, sr: float):
        x, y = seg[0], seg[1]
        if self.mode == "stft":
            f, t, X, Y = self._stft_xy(x, y, sr)
            X_mag, Y_mag = np.abs(X), np.abs(Y)
            if self.stft_power == 2.0:
                X_map, Y_map = X_mag**2, Y_mag**2
            elif self.stft_power == 1.0:
                X_map, Y_map = X_mag, Y_mag
            else:
                X_map, Y_map = X_mag**self.stft_power, Y_mag**self.stft_power
            chans = [X_map, Y_map]
        elif self.mode == "stft+cross":
            f, t, X, Y = self._stft_xy(x, y, sr)
            X_mag, Y_mag = np.abs(X), np.abs(Y)
            if self.stft_power == 2.0:
                X_map, Y_map = X_mag**2, Y_mag**2
            elif self.stft_power == 1.0:
                X_map, Y_map = X_mag, Y_mag
            else:
                X_map, Y_map = X_mag**self.stft_power, Y_mag**self.stft_power
            XY = X * np.conj(Y)
            cross_mag = np.abs(XY)
            phase_cos = np.cos(np.angle(XY))
            chans = [X_map, Y_map, cross_mag, phase_cos]
        else:  # "stft_complex"
            z = x + 1j*y
            pad = (self.stft_nperseg // 2) if self.stft_center else 0
            if pad > 0:
                z = np.pad(z, (pad, pad), mode="reflect")
            win = get_window(self.stft_window, self.stft_nperseg, fftbins=True)
            noverlap = self.stft_nperseg - self.stft_hop
            f, t, Z = stft(z, fs=sr, window=win, nperseg=self.stft_nperseg,
                           noverlap=noverlap, nfft=None, padded=False, boundary=None)
            amp = np.abs(Z)
            phase = np.angle(Z)
            phase_cos = np.cos(phase)
            phase_sin = np.sin(phase)
            phase_dev90 = phase - (np.pi/2)
            chans = [amp, phase_cos, phase_sin, phase_dev90]
        arr = np.stack(chans, axis=0)  # (C, F, T)
        return f, t, arr

    # ---------- CWT ----------
    def _cwt_xy(self, x: np.ndarray, y: np.ndarray, sr: float):
        scales = self.cwt_scale_base ** np.linspace(
            np.log(2)/np.log(self.cwt_scale_base),
            np.log(self.cwt_num_scales+1)/np.log(self.cwt_scale_base),
            self.cwt_num_scales
        ).astype(np.float32)
        Wx, fx = pywt.cwt(x, scales, self.cwt_wavelet, sampling_period=1.0/sr)
        Wy, fy = pywt.cwt(y, scales, self.cwt_wavelet, sampling_period=1.0/sr)
        # fx, fy는 동일해야 함
        return scales, fx, Wx, Wy

    def _build_cwt_maps(self, seg: np.ndarray, sr: float):
        x, y = seg[0], seg[1]
        if self.mode == "cwt":
            _, f, Wx, Wy = self._cwt_xy(x, y, sr)
            Xabs, Yabs = np.abs(Wx), np.abs(Wy)
            chans = [Xabs, Yabs]
        elif self.mode == "cwt+cross":
            _, f, Wx, Wy = self._cwt_xy(x, y, sr)
            Xabs, Yabs = np.abs(Wx), np.abs(Wy)
            Wxy = Wx * np.conj(Wy)
            cross_abs = np.abs(Wxy)
            phase_cos = np.cos(np.angle(Wxy))
            chans = [Xabs, Yabs, cross_abs, phase_cos]
        else:  # "cwt_complex"
            z = x + 1j*y
            scales = self.cwt_scale_base ** np.linspace(
                np.log(2)/np.log(self.cwt_scale_base),
                np.log(self.cwt_num_scales+1)/np.log(self.cwt_scale_base),
                self.cwt_num_scales
            ).astype(np.float32)
            Wz, f = pywt.cwt(z, scales, self.cwt_wavelet, sampling_period=1.0/sr)
            amp = np.abs(Wz)
            phase = np.angle(Wz)
            phase_cos = np.cos(phase)
            phase_sin = np.sin(phase)
            phase_dev90 = phase - (np.pi/2)
            chans = [amp, phase_cos, phase_sin, phase_dev90]
        arr = np.stack(chans, axis=0)  # (C, S, T)
        return f, arr

    # ---------- 호출 ----------
    def __call__(self, seg: np.ndarray, sr: float, rpm: float) -> torch.Tensor:
        f_rot = float(rpm) / 60.0  # 회전주파수 [Hz]
        if f_rot <= 0:
            raise ValueError("rpm must be > 0 for order normalization.")
        # 1) 계수/채널 맵 + 주파수축 얻기
        if self.mode.startswith("stft"):
            f, t, arr = self._build_stft_maps(seg, sr)  # arr: (C, F, T)
            order = f / f_rot                            # (F,)
        else:
            f, arr = self._build_cwt_maps(seg, sr)      # arr: (C, S, T)
            order = f / f_rot                            # (S,)
        # 2) order 마스킹 (0 < order ≤ max_order)
        mask = (order > 0) & (order <= self.max_order)
        if not np.any(mask):
            # 모든 bin이 마스크되면 최소한 한 줄은 유지
            idx = np.argmax(order > 0)
            mask = np.zeros_like(order, dtype=bool)
            mask[idx] = True
        arr = arr[:, mask, :]           # (C, Hm, T)
        order = order[mask]             # (Hm,)
        # 3) order축을 0..max_order 균일 그리드로 리샘플
        #    현재 arr는 order가 불균일일 수 있으므로, 채널/시간별로 1D 보간
        Hm, T = arr.shape[1], arr.shape[2]
        order_target = np.linspace(0.0, self.max_order, self.H_out, endpoint=True)
        out_ord = np.empty((arr.shape[0], self.H_out, T), dtype=arr.dtype)
        # order가 단조 증가임을 가정 (STFT/CWT에서 주파수 증가는 보장)
        for c in range(arr.shape[0]):
            for t_idx in range(T):
                out_ord[c, :, t_idx] = np.interp(order_target, order, arr[c, :, t_idx],
                                                 left=arr[c, 0, t_idx], right=arr[c, -1, t_idx])
        # 4) 시간축도 고정 bins로 리사이즈
        out = self._resize_CHW(out_ord, self.H_out, self.W_out)  # (C, H_out, W_out)
        # 5) log/정규화 → 텐서
        out = self._apply_log_norm(out)
        return torch.as_tensor(out, dtype=self.out_dtype)    

def _channel_labels_for_mode(mode: str):
    mode = mode.lower()
    if mode == "stft":
        return ["|X|^p", "|Y|^p"]
    if mode == "stft+cross":
        return ["|X|^p", "|Y|^p", "|X·Y*|", "cos(Δφ)"]
    if mode == "stft_complex":
        return ["|Z|", "cos(∠Z)", "sin(∠Z)", "∠Z − 90°"]
    if mode == "cwt":
        return ["|Wx|", "|Wy|"]
    if mode == "cwt+cross":
        return ["|Wx|", "|Wy|", "|Wx·Wy*|", "cos(Δφ)"]
    if mode == "cwt_complex":
        return ["|Wz|", "cos(∠Wz)", "sin(∠Wz)", "∠Wz − 90°"]
    # fallback
    return [f"ch{i}" for i in range(16)]

def visualize_imaging_tensor(
    tensor_chw,              # torch.Tensor or np.ndarray, shape (C,H,W)
    mode: str,
    max_order: float,
    window_sec: float,
    save_path: str | None = None,
    figsize=(18, 18),
    percent_clip=(2, 98),    # magnitude 계열 대비 향상용 퍼센타일 클리핑
):
    """
    단일 샘플 시각화: 채널들을 가로/세로 그리드로 출력
    - y축: Order (0 .. max_order)
    - x축: Time (0 .. window_sec)
    """
    # to numpy (C,H,W)
    arr = tensor_chw.detach().cpu().numpy() if hasattr(tensor_chw, "detach") else np.asarray(tensor_chw)
    assert arr.ndim == 3, "Input must be (C,H,W)"
    C, H, W = arr.shape

    ch_labels = _channel_labels_for_mode(mode)
    if len(ch_labels) < C:
        ch_labels += [f"ch{i}" for i in range(len(ch_labels), C)]

    # subplot grid 크기 잡기
    ncols = min(4, C)
    nrows = int(np.ceil(C / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    extent = [0.0, float(window_sec), 0.0, float(max_order)]  # x: time(sec), y: order
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx < C:
                img = arr[idx]

                # 채널 유형별 vmin/vmax 설정
                label = ch_labels[idx].lower()
                if any(k in label for k in ["cos(", "sin("]):
                    vmin, vmax = -1.0, 1.0
                elif "− 90°" in ch_labels[idx] or "- 90°" in ch_labels[idx] or "90" in ch_labels[idx]:
                    # 위상 편차 채널: 대략 -π..π
                    vmin, vmax = -np.pi, np.pi
                else:
                    # magnitude 계열: 퍼센타일 클리핑
                    lo, hi = np.percentile(img, percent_clip)
                    vmin, vmax = float(lo), float(hi) if hi > lo else (None, None)

                im = ax.imshow(
                    img,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    vmin=vmin, vmax=vmax,
                    cmap="magma",
                )
                ax.set_title(ch_labels[idx], fontsize=10)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Order")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis("off")
            idx += 1

    fig.suptitle(f"Mode: {mode} | (C,H,W)=({C},{H},{W})", fontsize=12)
    fig.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig) if save_path else plt.show()


if __name__ == '__main__':
    data_root = os.path.join(os.getcwd(), 'processed')
    meta_csv = os.path.join(data_root, 'meta.csv')
    meta_pd = pd.read_csv(meta_csv)

    meta_pd['sensor_position'] = meta_pd['sensor_position'].apply(ast.literal_eval)
    meta_pd = meta_pd[5 <= meta_pd['data_sec']]
    
    # 필요한 데이터셋만 남기고 나머지는 제거
    using_dataset = ['dxai', 'iis', 'vat', 'vbl', 'mfd']
    meta_pd = meta_pd[meta_pd['dataset'].isin(using_dataset)]
    print(f'Filtered metadata to {len(meta_pd)} rows with datasets in {using_dataset}')

    signal_imger = OrderInvariantSignalImager(
        mode='cwt+cross',
        log1p=True,
        normalize="per_channel",  # None | "per_channel" | "global"
        eps=1e-8,
        out_dtype=torch.float32,
        max_order=20.0,           # order 축 상한
        H_out=256,                # order-bin 수
        W_out=256,                # time-bin 수
        # STFT
        stft_nperseg=1024,
        stft_hop=256,
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
    t0, y, info = dataset[100]  # (C,H,W)
    print(info)
    visualize_imaging_tensor(t0, mode="cwt+cross", max_order=20, window_sec=5.0, save_path='test')