import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
from cupyx.scipy.signal import filtfilt, iirnotch
from cupyx.scipy.signal import spectrogram as cupyx_spectrogram
from scipy.signal import filtfilt as scipy_filtfilt, butter as scipy_butter

def create_train_data(path):
    CLASSES = [
        "seizure_vote", "lpd_vote", "gpd_vote", 
        "lrda_vote", "grda_vote", "other_vote"
    ]
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    
    # Create a new identifier combining multiple columns
    id_cols = [
        'eeg_id', 'spectrogram_id', 'seizure_vote', 'lpd_vote', 
        'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'
    ]
    df['new_id'] = df[id_cols].astype(str).agg('_'.join, axis=1)
    
    # Calculate the sum of votes for each class
    df['sum_votes'] = df[CLASSES].sum(axis=1)
    
    # Group the data by the new identifier and aggregate various features
    agg_functions = {
        'eeg_id': 'first',
        'eeg_label_offset_seconds': ['min', 'max'],
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'spectrogram_id': 'first',
        'patient_id': 'first',
        'expert_consensus': 'first',
        **{col: 'sum' for col in CLASSES},
        'sum_votes': 'mean',
    }
    grouped_df = df.groupby('new_id').agg(agg_functions).reset_index()

    # Flatten the MultiIndex columns and adjust column names
    grouped_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped_df.columns]
    grouped_df.columns = grouped_df.columns.str.replace('_first', '').str.replace('_sum', '').str.replace('_mean', '')
    
    return grouped_df


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
    sci_bandpass_coefficients = scipy_butter(order_band, 
                                             [low_cut_freq_normalized, high_cut_freq_normalized],
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


def create_spectogram_competition(spec_id, seconds_min):
    spec = pd.read_parquet(f'./hms/data/train_spectrograms/{spec_id}.parquet')
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


def process_eegs(train_df, output_folder):
    """Process EEGs and save the final images."""
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the EEG data frame
    for i in tqdm(range(len(train_df)), desc="Processing EEGs"):
        row = train_df.iloc[i]
        eeg_id = row['eeg_id']
        spec_id = row['spectrogram_id']
        seconds_min = int(row['spectrogram_label_offset_seconds_min'])
        start_second = int(row['eeg_label_offset_seconds_min'])
        
        # Load EEG data from file
        eeg_data = pd.read_parquet(f'./hms/data/train_eegs/{eeg_id}.parquet')
        eeg_new_key = f'{eeg_id}_{seconds_min}_{start_second}'

        # Generate spectrogram images from EEG data
        image_50s, _ = create_spectrogram_with_cupy(eeg_data=eeg_data, eeg_id=eeg_id, start=start_second, 
                                                    duration= 50, low_cut_freq = 0.7, high_cut_freq = 20, 
                                                    order_band = 5, spec_size_freq = 267, spec_size_time = 501,
                                                    nperseg = 1500, noverlap = 1483, nfft = 2750,
                                                    sigma_gaussian = 0.0, mean_montage_names = 4)
        image_10s, _ = create_spectrogram_with_cupy(eeg_data=eeg_data, eeg_id=eeg_id, start=start_second, 
                                                    duration= 10, low_cut_freq = 0.7, high_cut_freq = 20, 
                                                    order_band = 5, spec_size_freq = 100, spec_size_time = 291,
                                                    nperseg = 260, noverlap = 254, nfft = 1030,
                                                    sigma_gaussian = 0.0, 
                                                    mean_montage_names = 4)
        
        image_10m = create_spectogram_competition(spec_id, seconds_min)
        
        # Save the final image in compressed format
        file_path = os.path.join(output_folder, f'{eeg_new_key}.npz')
        np.savez_compressed(file_path, image_50s=image_50s, image_10s=image_10s, image_10m=image_10m)


if __name__ == '__main__':
    train_csv_path = './hms'
    output_folder = './hms/train_data'
    train_df = create_train_data(train_csv_path)
    print(train_df)
    process_eegs(train_df, output_folder)