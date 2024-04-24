import sys
import os
import gc
import copy
import yaml
import random
import shutil
from time import time
import typing as tp
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

import timm
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from cupyx.scipy.signal import filtfilt, iirnotch
from cupyx.scipy.signal import spectrogram as cupyx_spectrogram
from scipy.signal import filtfilt as scipy_filtfilt, butter as scipy_butter


class CFG:
    in_h = 800
    in_w = 800
    in_channels = 1
    batch_size = 32
    deterministic = True
    device = "cuda"
    model_name = 'efficientnet_b2'
    model_path_below10 = '/kaggle/input/20240406_effib2_800_gsn0.5_cv0.602/pytorch/20240406_effib2_800_gsn0.5_cv0.602/1'
    model_path_above10 = '/kaggle/input/20240405-effib2-800-gsn0-5-above10-cv0-337'


## config
ROOT = '/kaggle/input/hms-harmful-brain-activity-classification'
TEST_SPEC = os.path.join(ROOT, "test_spectrograms")
TEST_EEG = os.path.join(ROOT, "test_eegs")

PWD_ROOT = Path.cwd().parent
TMP = os.path.join(PWD_ROOT, "tmp")
TEST_SPEC_SPLIT = os.path.join(TMP, "test_spectrograms_split")

if not os.path.exists(TEST_SPEC_SPLIT):
    os.makedirs(TEST_SPEC_SPLIT)
    
TEST_EEG_SPEC_SPLIT = os.path.join(TMP, "test_eeg_spec_split")

if not os.path.exists(TEST_EEG_SPEC_SPLIT):
    os.makedirs(TEST_EEG_SPEC_SPLIT)

RANDAM_SEED = 1086
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
N_CLASSES = len(CLASSES)
FOLDS = [0, 1, 2, 3, 4]
N_FOLDS = len(FOLDS)


## Read data
test = pd.read_csv(os.path.join(ROOT, "test.csv"))
print(f'test csv:\n{test.head()}')


def create_spectrogram_with_cupy(eeg_data, eeg_id, start, duration=50,
                                 low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                 spec_size_freq=267, spec_size_time=30,
                                 nperseg=1500, noverlap=1483, nfft=2750,
                                 sigma_gaussian=0.7,
                                 mean_montage_names=4):
    electrode_pair_name_locations = {'LL': ['Fp1', 'F7', 'T3', 'T5', 'O1'],
                                     'RL': ['Fp2', 'F8', 'T4', 'T6', 'O2'],
                                     'LP': ['Fp1', 'F3', 'C3', 'P3', 'O1'],
                                     'RP': ['Fp2', 'F4', 'C4', 'P4', 'O2']}

    # Filter specifications
    nyquist_freq = 0.5 * 200
    low_cut_freq_normalized = low_cut_freq / nyquist_freq
    high_cut_freq_normalized = high_cut_freq / nyquist_freq

    # Bandpass and notch filter
    notch_coefficients = iirnotch(w0=60, Q=30, fs=200)
    sci_bandpass_coefficients = scipy_butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized],
                                             btype='band')

    spec_size = duration * 200
    start = start * 200
    real_start = start + (10_000 // 2) - (spec_size // 2)
    eeg_data = eeg_data.iloc[real_start:real_start + spec_size]

    # Spectrogram parameters
    fs = 200

    if spec_size_freq <= 0 or spec_size_time <= 0:
        freq_size = int((nfft // 2) / 5.15198) + 1
        segments = int((spec_size - noverlap) / (nperseg - noverlap))
    else:
        freq_size = spec_size_freq
        segments = spec_size_time

    # Initialize spectrogram container
    spectrogram = cp.zeros((freq_size, segments, 4), dtype='float32')

    processed_eeg = {}

    for i, (electrode_pair_name, electrode_locs) in enumerate(electrode_pair_name_locations.items()):
        processed_eeg[electrode_pair_name] = np.zeros(spec_size)

        for j in range(4):
            # Compute differential signals
            signal = cp.array(eeg_data[electrode_locs[j]].values - eeg_data[electrode_locs[j + 1]].values)

            # Handles NaNs 
            mean_signal = cp.nanmean(signal)
            signal = cp.nan_to_num(signal, nan=mean_signal) if cp.isnan(signal).mean() < 1 else cp.zeros_like(signal)

            # Filters bandpass and notch
            signal_filtered = filtfilt(*notch_coefficients, signal)
            signal_filtered = scipy_filtfilt(*sci_bandpass_coefficients, signal_filtered.get())  # HOTFIX

            # GPU-accelerated spectrogram computation
            frequencies, times, Sxx = cupyx_spectrogram(signal_filtered, fs, nperseg=nperseg, noverlap=noverlap,
                                                        nfft=nfft)

            # Filters frequency range 
            valid_freq = (frequencies >= 0.59) & (frequencies <= 20)
            Sxx_filtered = Sxx[valid_freq, :]

            # Logarithmic transformation and normalization using Cupy
            spectrogram_slice = cp.clip(Sxx_filtered, cp.exp(-4), cp.exp(6))
            spectrogram_slice = cp.log10(spectrogram_slice)

            normalization_epsilon = 1e-6
            mean = spectrogram_slice.mean(axis=(0, 1), keepdims=True)
            std = spectrogram_slice.std(axis=(0, 1), keepdims=True)
            spectrogram_slice = (spectrogram_slice - mean) / (std + normalization_epsilon)

            spectrogram[:, :, i] += spectrogram_slice
            processed_eeg[f'{electrode_locs[j]}_{electrode_locs[j + 1]}'] = signal.get()
            processed_eeg[electrode_pair_name] += signal.get()

        # AVERAGES THE 4 MONTAGE DIFFERENCES
        if mean_montage_names > 0:
            spectrogram[:, :, i] /= mean_montage_names

    # Applies Gaussian filter and retrieves the spectrogram as a NumPy array using cupy.ndarray.get()
    spec_numpy = gaussian_filter(spectrogram, sigma=sigma_gaussian).get() if sigma_gaussian > 0 else spectrogram.get()

    # Filter EKG signal
    ekg_signal_filtered = filtfilt(*notch_coefficients, cp.array(eeg_data["EKG"].values))
    processed_eeg['EKG'] = scipy_filtfilt(*sci_bandpass_coefficients, ekg_signal_filtered.get())  # HOTFIX
    return spec_numpy, processed_eeg


def create_spectogram_competition(spec_id, seconds_min=0):
    spec = pd.read_parquet(f'{TEST_SPEC}/{spec_id}.parquet')
    inicio = (seconds_min) // 2
    img = spec.fillna(0).values[:, 1:].T.astype("float32")
    img = img[:, inicio:inicio+300]
    
    # Log transform and normalize
    img = np.clip(img, np.exp(-4), np.exp(6))
    img = np.log(img)
    eps = 1e-6
    img_mean = img.mean()
    img_std = img.std()
    img = (img - img_mean) / (img_std + eps)
    
    return img

def process_eegs(eeg_id, spec_id):
    """Process EEGs and save the final images."""
    # Load EEG data from file
    eeg_data = pd.read_parquet(os.path.join(TEST_EEG, f'{eeg_id}.parquet'))

    # Generate spectrogram images from EEG data
    image_50s, _ = create_spectrogram_with_cupy(eeg_data=eeg_data, eeg_id=eeg_id, start=0, 
                                                duration= 50, low_cut_freq = 0.7, high_cut_freq = 20, 
                                                order_band = 5, spec_size_freq = 267, spec_size_time = 501,
                                                nperseg = 1500, noverlap = 1483, nfft = 2750,
                                                sigma_gaussian = 0.0, 
                                                mean_montage_names = 4)
    image_10s, _ = create_spectrogram_with_cupy(eeg_data=eeg_data, eeg_id=eeg_id, start=0, 
                                                duration= 10, low_cut_freq = 0.7, high_cut_freq = 20, 
                                                order_band = 5, spec_size_freq = 100, spec_size_time = 291,
                                                nperseg = 260, noverlap = 254, nfft = 1030,
                                                sigma_gaussian = 0.0, 
                                                mean_montage_names = 4)

    image_10m = create_spectogram_competition(spec_id, seconds_min=0)

    # Save the final image in compressed format
    file_path = os.path.join(TEST_SPEC_SPLIT, f'{eeg_id}.npz')
    np.savez_compressed(file_path, image_50s=image_50s, image_10s=image_10s, image_10m=image_10m)
        
test = test.rename({'spectrogram_id':'spec_id'},axis=1)
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

for i in range(len(test)):
    row = test.iloc[i]
    eeg_id = row['eeg_id']
    spec_id = row['spec_id']
    process_eegs(eeg_id, spec_id)


## define dataset
FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]

class HMSHBACSpecDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths: tp.Sequence[FilePath],
                 labels: tp.Sequence[Label],
                 transform: A.Compose,
                 in_chl: int=1):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.in_chl = in_chl

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img = np.load(img_path)
        
        img_50s = img['image_50s']
        img_10s = img['image_10s']
        img_10m = img['image_10m']

        _img = np.zeros((800, 800), dtype=np.float32)
        for i in range(4):
            _img[i*200:(i+1)*200,:500] = img_50s[:200,:500, i]
            _img[i*100:(i+1)*100,500:500+291] = img_10s[:, :, i]
            _img[400+i*100:400+(i+1)*100,500:] = img_10m[i*100:(i+1)*100, :]

        _img = self._apply_transform(_img)

        return {"data": _img, "target": label}


    def _apply_transform(self, img: np.ndarray):
        """apply transform to image and mask"""
        transformed = self.transform(image=img)
        img = transformed["image"]
        return img


def get_transforms(CFG):
    test_transform = A.Compose([
        ToTensorV2(p=1.0)
    ])
    return test_transform


## define model
class HMSHBACSpecModel(nn.Module):
    def __init__(self,
                 model_name: str,
                 pretrained: bool,
                 in_channels: int,
                 num_classes: int,):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels)

    def forward(self, x):
        h = self.model(x)
        return h


## train config
def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore
    
def to_device(tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
              device: torch.device, *args, **kwargs):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)
    
def get_test_path_label(test: pd.DataFrame):
    """Get file path and target info."""

    img_paths = []
    eeg_paths = []
    labels = np.full((len(test), 6), -1, dtype=np.float32)
        
    for test_info in test.groupby("spec_id"):
        spec_id = test_info[1]['spec_id'].values[0]
        eeg_id = test_info[1]['eeg_id'].values[0]

        img_path = os.path.join(TEST_SPEC_SPLIT, f"{eeg_id}.npz")
        img_paths.append(img_path)
        
    test_data = {
        "image_paths": img_paths,
        "labels": [l for l in labels]}
    
    return test_data


## inference
def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())
        
    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr

test_pred_arr_below10 = np.zeros((N_FOLDS, len(test), N_CLASSES))
test_pred_arr_above10 = np.zeros((N_FOLDS, len(test), N_CLASSES))

## get_dataloader
test_path_label = get_test_path_label(test)
test_transform = get_transforms(CFG)
test_dataset = HMSHBACSpecDataset(**test_path_label, transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=CFG.batch_size, num_workers=4, shuffle=False, drop_last=False)

device = torch.device(CFG.device)

for fold_id in range(N_FOLDS):
    print(f"\n[fold {fold_id}]")
    
    # # get model
    model_path_above10 = os.path.join(CFG.model_path_above10, f'snapshot_{fold_id}_best_model.pth')
    model_above10 = HMSHBACSpecModel(model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels)
    model_above10.load_state_dict(torch.load(model_path_above10, map_location=device))
    
    model_path_below10 = os.path.join(CFG.model_path_below10, f'snapshot_{fold_id}_best_model.pth')
    model_below10 = HMSHBACSpecModel(model_name=CFG.model_name, pretrained=False, num_classes=6, in_channels=CFG.in_channels)
    model_below10.load_state_dict(torch.load(model_path_below10, map_location=device))
    
    # # inference
    val_pred_above10 = run_inference_loop(model_above10, test_loader, device)
    test_pred_arr_above10[fold_id] = val_pred_above10
    
    val_pred_below10 = run_inference_loop(model_below10, test_loader, device)
    test_pred_arr_below10[fold_id] = val_pred_below10
    
    del model_path_above10
    del model_path_below10
    torch.cuda.empty_cache()
    gc.collect()

test_pred_arr_above10 = test_pred_arr_above10.mean(axis=0)
test_pred_arr_below10 = test_pred_arr_below10.mean(axis=0)

test_pred_arr = test_pred_arr_above10 * 0.8 + test_pred_arr_below10 * 0.2


## submit
test_pred_df = pd.DataFrame(test_pred_arr, columns=CLASSES)
test_pred_df = pd.concat([test[["eeg_id"]], test_pred_df], axis=1)
test_pred_df.to_csv('submission.csv', index=False)
test_pred_df.head()