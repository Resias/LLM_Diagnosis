import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window
from scipy.signal import decimate
from scipy.stats import kurtosis

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
        self, data_root, window_sec, stride_sec,
        using_dataset = ['dxai', 'iis', 'vat', 'vbl', 'mfd'],
        drop_last=True, dtype=torch.float32, transform=None,
        channel_order=("x","y"), cache_mode="file",  # 'none' | 'file' | 'windows'
    ):
        
        meta_csv = os.path.join(data_root, 'meta.csv')
        meta_pd = pd.read_csv(meta_csv)

        meta_pd['sensor_position'] = meta_pd['sensor_position'].apply(ast.literal_eval)
        meta_pd = meta_pd[5 <= meta_pd['data_sec']]
        meta_pd = meta_pd[meta_pd['dataset'].isin(using_dataset)]
        meta_pd['class_name'].unique()
        
        self.meta_df = meta_pd.reset_index(drop=True).copy()
        
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
        self.include_normal_ref = True  # always return a normal-reference sample from the same dataset & load_condition

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

        # ---- Build normal-reference window pool: key = (dataset, load_condition) ----
        self._normal_pool = {}  # (dataset, load_condition) -> list of (row_idx, start)
        for n_row_idx, n_row in self.meta_df.iterrows():
            if self.meta_df.loc[n_row_idx, 'merged_class'] != 'normal':
                continue
            sr_n = float(n_row["sampling_rate"])
            win_n_n = int(round(self.window_sec * sr_n))
            stride_n_n = int(round(self.stride_sec * sr_n))
            starts_n = _starts_for(np.load(os.path.join(self.data_root, n_row["file_name"])).shape[1],
                                   win_n_n, stride_n_n, self.drop_last)
            key = (n_row["dataset"], n_row["load_condition"])
            if key not in self._normal_pool:
                self._normal_pool[key] = []
            for s_n in starts_n:
                self._normal_pool[key].append((n_row_idx, s_n))
        # fallback pool by dataset only (if no matching load_condition exists)
        self._normal_pool_by_ds = {}
        for n_row_idx, n_row in self.meta_df.iterrows():
            if self.meta_df.loc[n_row_idx, 'merged_class'] != 'normal':
                continue
            sr_n = float(n_row["sampling_rate"])
            win_n_n = int(round(self.window_sec * sr_n))
            stride_n_n = int(round(self.stride_sec * sr_n))
            starts_n = _starts_for(np.load(os.path.join(self.data_root, n_row["file_name"])).shape[1],
                                   win_n_n, stride_n_n, self.drop_last)
            key_ds = (n_row["dataset"],)
            if key_ds not in self._normal_pool_by_ds:
                self._normal_pool_by_ds[key_ds] = []
            for s_n in starts_n:
                self._normal_pool_by_ds[key_ds].append((n_row_idx, s_n))

    def _extract_segment(self, row_idx, start):
        """Return (seg ndarray shape (2, win_n)) for the given row & start, respecting cache_mode."""
        row = self.meta_df.iloc[row_idx]
        meta = self._row_meta[row_idx]
        sr, x_idx, y_idx, win_n = meta["sr"], meta["x_idx"], meta["y_idx"], meta["win_n"]
        if self.cache_mode == "none":
            arr = np.load(meta["file_path"], mmap_mode="r")
            x_seg = arr[x_idx, start:start+win_n]
            y_seg = arr[y_idx, start:start+win_n]
        elif self.cache_mode == "file":
            base = self._file_cache[row_idx]
            x_seg = base[x_idx, start:start+win_n]
            y_seg = base[y_idx, start:start+win_n]
        else:  # 'windows'
            stride_n = int(round(self.stride_sec * sr))
            widx = (start // stride_n)
            win = self._win_cache[row_idx]  # (W, 2, win_n)
            seg = win[widx]
            # ensure correct channel order
            if self.channel_order == ("x","y"):
                return seg
            else:
                return seg[::-1]
        seg = np.stack([x_seg, y_seg], axis=0) if self.channel_order==("x","y") \
              else np.stack([y_seg, x_seg], axis=0)
        return seg

    def _pick_normal_reference(self, dataset, load_condition):
        """Return (row_idx, start) of a normal sample matching (dataset, load_condition) if available,
        otherwise same dataset only, otherwise None."""
        key = (dataset, load_condition)
        pool = self._normal_pool.get(key)
        if pool:
            ridx, s = pool[np.random.randint(len(pool))]
            return ridx, s
        # fallback by dataset only
        key_ds = (dataset,)
        pool2 = self._normal_pool_by_ds.get(key_ds)
        if pool2:
            ridx, s = pool2[np.random.randint(len(pool2))]
            return ridx, s
        return None

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx, data_info=True):
        row_idx, start = self.index_map[idx]
        row = self.meta_df.iloc[row_idx]
        meta = self._row_meta[row_idx]
        sr = meta["sr"]

        # ---- current segment ----
        seg = self._extract_segment(row_idx, start)
        tensor_img = self.transform(seg, sr=sr, rpm=float(row["rpm"]))
        class_idx = self.class_list.index(row['merged_class'])
        tensor_cls = torch.tensor(class_idx ,dtype=torch.long)

        info = {
            "sampling_rate": float(sr),
            "rpm": float(row["rpm"]),
            "label_class": str(row["class_name"]),
            "merged_class": str(row["merged_class"]),
            "severity": str(row["severity"]),
            "load_condition": str(row["load_condition"]),
            "dataset": str(row["dataset"]),
            "file_name": str(row["file_name"]),
        }

        # ---- normal reference (same dataset & load_condition) ----
        normal_tuple = None
        if self.include_normal_ref:
            pick = self._pick_normal_reference(row["dataset"], row["load_condition"])
            if pick is not None:
                n_row_idx, n_start = pick
                n_row = self.meta_df.iloc[n_row_idx]
                n_meta = self._row_meta[n_row_idx]
                n_sr = n_meta["sr"]
                n_seg = self._extract_segment(n_row_idx, n_start)
                tensor_img_norm = self.transform(n_seg, sr=n_sr, rpm=float(n_row["rpm"]))
                tensor_cls_norm = torch.tensor(self.class_list.index('normal'), dtype=torch.long)
                n_info = {
                    "sampling_rate": float(n_sr),
                    "rpm": float(n_row["rpm"]),
                    "label_class": str(n_row["class_name"]),
                    "merged_class": str(n_row["merged_class"]),
                    "severity": str(n_row["severity"]),
                    "load_condition": str(n_row["load_condition"]),
                    "dataset": str(n_row["dataset"]),
                    "file_name": str(n_row["file_name"]),
                }
                normal_tuple = (tensor_img_norm, tensor_cls_norm, n_info)

        if not data_info:
            if normal_tuple is None:
                return tensor_img, tensor_cls
            else:
                tensor_img_norm, tensor_cls_norm, _ = normal_tuple
                return tensor_img, tensor_cls, tensor_img_norm, tensor_cls_norm

        # ---- knowledge strings when data_info=True ----
        cur_kn = _build_knowledge_string(seg, sr=float(sr), rpm=float(row["rpm"]))
        info["knowledge"] = cur_kn
        if normal_tuple is not None:
            _, _, n_info = normal_tuple
            n_kn = _build_knowledge_string(n_seg, sr=float(n_sr), rpm=float(n_row["rpm"]))
            n_info["knowledge"] = n_kn
            return tensor_img, tensor_cls, info, normal_tuple[0], normal_tuple[1], n_info
        else:
            return tensor_img, tensor_cls, info

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
    ):
        assert mode in {
            "stft", "stft+cross", "stft_complex",
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
            print('err')
            exit()
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

# ---- Helper for knowledge string ----
def _dominant_order_peaks(seg, sr, rpm, k=3, tol=0.15, ds_factor=4, seg_factor=4, sec_win=1.0):
    """
    Compute dominant spectral peaks using ensemble averaging to reduce noise.
    Steps:
      1) Optional downsampling by ds_factor (default 4).
      2) Split the signal into `seg_factor` contiguous segments (default 4).
      3) Within each segment, cut into non-overlapping windows of `sec_win` seconds (default 1.0s).
      4) Compute Hann-windowed rFFT magnitude of each window and average across all windows.
      5) Return top-k peak frequencies and their proximity to 1x/2x/3x orders.

    Returns: dict with
      - peaks_hz: list of top-k peak frequencies (Hz)
      - orders: list of their order values (peak_hz / f_rot)
      - near_1x / near_2x / near_3x: boolean flags
    """
    x = np.asarray(seg[0], dtype=float)
    if x.size == 0 or sr <= 0 or rpm <= 0:
        return {"peaks_hz": [], "orders": [], "near_1x": False, "near_2x": False, "near_3x": False}

    # 1) Downsample (zero-phase IIR decimation to suppress aliasing)
    sr_eff = float(sr)
    if ds_factor and ds_factor > 1:
        try:
            x = decimate(x, ds_factor, ftype='iir', zero_phase=True)
            sr_eff = sr_eff / ds_factor
        except Exception:
            # Fallback: naive decimation if scipy version lacks zero_phase
            x = x[::ds_factor]
            sr_eff = sr_eff / ds_factor

    n_total = x.shape[0]
    if n_total < int(sec_win * sr_eff):
        # Not enough samples for a 1-second window; fall back to plain rFFT
        freqs = np.fft.rfftfreq(n_total, d=1.0/sr_eff)
        mag = np.abs(np.fft.rfft(x))
        if mag.size > 0:
            mag[0] = 0.0
    else:
        # 2) Split into seg_factor segments
        seg_factor = max(1, int(seg_factor))
        seg_len = n_total // seg_factor
        window_n = int(round(sec_win * sr_eff))
        if window_n <= 0: window_n = min(n_total, 1024)

        # 3) Within each segment, cut into non-overlapping 1s windows and accumulate spectra
        acc_mag = None
        count = 0
        hann = np.hanning(window_n)
        for s in range(seg_factor):
            start = s * seg_len
            end = start + seg_len if s < seg_factor - 1 else n_total
            seg_x = x[start:end]
            m = seg_x.shape[0] // window_n
            for i in range(m):
                w = seg_x[i*window_n:(i+1)*window_n]
                w = w - np.mean(w)
                W = np.fft.rfft(w * hann)
                mag_w = np.abs(W)
                if acc_mag is None:
                    acc_mag = mag_w
                else:
                    # Align length in case of minor rounding differences
                    L = min(acc_mag.shape[0], mag_w.shape[0])
                    acc_mag = acc_mag[:L] + mag_w[:L]
                count += 1
        if count == 0:
            # Fallback to whole-signal FFT
            freqs = np.fft.rfftfreq(n_total, d=1.0/sr_eff)
            mag = np.abs(np.fft.rfft(x))
        else:
            mag = acc_mag / float(count)
            freqs = np.fft.rfftfreq(window_n, d=1.0/sr_eff)
        if mag.size > 0:
            mag[0] = 0.0

    # 4) Pick top-k peaks by magnitude
    if mag.size == 0:
        return {"peaks_hz": [], "orders": [], "near_1x": False, "near_2x": False, "near_3x": False}
    k = max(1, int(k))
    idx = np.argsort(mag)[-k:]
    idx = idx[np.argsort(freqs[idx])]
    peaks_hz = freqs[idx].tolist()

    # 5) Map to orders and proximity checks
    f_rot = rpm / 60.0
    orders = [p / f_rot if f_rot > 0 else 0.0 for p in peaks_hz]
    def near(o, target): return abs(o - target) <= tol
    near_flags = {
        "near_1x": any(near(o, 1.0) for o in orders),
        "near_2x": any(near(o, 2.0) for o in orders),
        "near_3x": any(near(o, 3.0) for o in orders),
    }
    return {"peaks_hz": [float(p) for p in peaks_hz], "orders": [float(o) for o in orders], **near_flags}

def _build_knowledge_string(seg, sr, rpm):
    """
    Build a compact textual summary for RAG/LLM reasoning from a window segment.
    Includes RMS, kurtosis, crest factor, and dominant order peaks proximity.
    """
    x = seg[0]; y = seg[1]
    def _rms(a): return float(np.sqrt(np.mean(a**2) + 1e-12))
    def _crest(a):
        r = _rms(a)
        return float((np.max(np.abs(a)) / (r + 1e-12)) if r > 0 else 0.0)
    rms_x, rms_y = _rms(x), _rms(y)
    k_x = float(kurtosis(x, fisher=False, bias=False)) if x.size > 8 else 3.0
    k_y = float(kurtosis(y, fisher=False, bias=False)) if y.size > 8 else 3.0
    cf_x, cf_y = _crest(x), _crest(y)
    dom = _dominant_order_peaks(seg, sr, rpm, k=5, ds_factor=4, seg_factor=4, sec_win=1.0)
    parts = [
        f"sr={sr:.1f}Hz, rpm={rpm:.1f}, f_rot={rpm/60.0:.3f}Hz",
        f"RMS(x,y)=({rms_x:.4g},{rms_y:.4g})",
        f"Kurtosis(x,y)=({k_x:.3g},{k_y:.3g})",
        f"CrestFactor(x,y)=({cf_x:.3g},{cf_y:.3g})",
        f"TopPeaksHz={','.join(f'{p:.2f}' for p in dom['peaks_hz'])}",
        f"Orders={','.join(f'{o:.2f}' for o in dom['orders'])}",
        f"near(1x)={dom['near_1x']}, near(2x)={dom['near_2x']}, near(3x)={dom['near_3x']}",
    ]
    return " | ".join(parts)