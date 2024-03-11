import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import librosa
import os
import logging

# Constants
SAMPLE_RATE = 44100
TOTAL_FEATURES = 23

log_filename = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v2/processing_log.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, 'a'), logging.StreamHandler()])

hyperparameters_combinations = [
    {"window_size": 5, "step_size": 2.5, "expected_timesteps": 23, "lstm_neurons": 128, "epochs": 20, "batch_size": 32},
    {"window_size": 10, "step_size": 5, "expected_timesteps": 11, "lstm_neurons": 64, "epochs": 20, "batch_size": 32}
    # {"window_size": 15, "step_size": 7.5, "expected_timesteps": 7, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
]

def expected_npz_files_per_config(num_audio_files, window_size, step_size, expected_timesteps, files_to_append):
    # Given each audio file is 1 minute, calculate total length of audio for n_files_per_batch
    total_audio_length = files_to_append * 60  # 60 seconds per audio file
    num_samples_per_window = int(window_size * SAMPLE_RATE)
    step_samples = int(step_size * SAMPLE_RATE)
    total_windows = (total_audio_length * SAMPLE_RATE - num_samples_per_window) // step_samples + 1
    sequences_per_batch = ((total_windows - expected_timesteps) // expected_timesteps) + 1
    
    # Total batches needed to process all files
    total_batches = num_audio_files // files_to_append + (1 if num_audio_files % files_to_append > 0 else 0)
    
    # Total expected npz files equals sequences per batch times the total number of batches
    total_expected_npz_files = sequences_per_batch * total_batches
    return total_expected_npz_files

def create_model_directory(root_path, config):
    try:
        model_dir = os.path.join(root_path, f"model_{'_'.join(map(str, config.values()))}")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    except Exception as e:
        logging.error(f"Failed to create model directory: {e}")
        return None

def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        return audio
    except Exception as e:
        logging.error(f"Failed to load audio file {file_path}: {e}")
        return None
    
def extract_spectral_features(audio, sample_rate):# Spectral Features 
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

def extract_temporal_features(audio):# Temporal Features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

def extract_additional_features(audio, sample_rate):#  Additional features
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)

def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)    # Extract MFCCs
    mfccs_processed = np.mean(mfccs.T,axis=0)
    
    spectral_features = extract_spectral_features(audio, sample_rate)    # Extract additional features
    temporal_features = extract_temporal_features(audio)
    additional_features = extract_additional_features(audio, sample_rate)
    
    features = np.concatenate((mfccs_processed, spectral_features, temporal_features, additional_features))    # Combine all features
    return features

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []

    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
    return windows

def concatenate_and_process_files(paths, sample_rate, hyperparameters, files_to_append, save_dir):
    for config in hyperparameters:
        window_size = config["window_size"]
        step_size = config["step_size"]
        expected_timesteps = config["expected_timesteps"]
        lstm_neurons = config["lstm_neurons"]  # Ensure these are used if needed
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        dataset_dirname = f"ws{window_size}_ss{step_size}_et{expected_timesteps}_lstm{lstm_neurons}_{epochs}epochs_bs{batch_size}"
        data_save_dir = os.path.join(save_dir, dataset_dirname,'data')
        os.makedirs(data_save_dir, exist_ok=True)

        actual_npz_count = 0  # Initialize actual npz file count

        for label, path in paths.items():
            file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
            num_audio_files = len(file_paths)  # Total number of audio files in the current path
            num_batches = len(file_paths) // files_to_append + (1 if len(file_paths) % files_to_append > 0 else 0)
            expected_npz_count = expected_npz_files_per_config(num_audio_files, window_size, step_size, expected_timesteps, files_to_append)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * files_to_append
                end_idx = min(start_idx + files_to_append, len(file_paths))
                batch_file_paths = file_paths[start_idx:end_idx]

                concatenated_audio = np.concatenate([load_audio_file(fp, sample_rate) for fp in batch_file_paths if load_audio_file(fp, sample_rate) is not None], axis=0)

                windows = sliding_window(concatenated_audio, window_size, step_size, sample_rate)
                for i in range(0, len(windows), expected_timesteps):
                    sequence = windows[i:i + expected_timesteps]
                    if len(sequence) == expected_timesteps:
                        features = np.array([extract_features(w, sample_rate) for w in sequence])
                        save_path = os.path.join(data_save_dir, f"{label}_config{config['window_size']}_batch{batch_idx}_seq{i}.npz")
                        np.savez_compressed(save_path, features=features)
                        actual_npz_count += 1  # Increment the actual npz file count

        # Log the number of npz files created for this configuration
        logging.info(f"Configuration {config}: Expected {expected_npz_count} .npz files, actual {actual_npz_count} .npz files created.")
                        
def save_features_with_metadata(paths, sample_rate, hyperparameters, files_to_append, save_dir):
    for config in hyperparameters:
        # Configuration-specific parameters
        window_size = config["window_size"]
        step_size = config["step_size"]
        expected_timesteps = config["expected_timesteps"]
        lstm_neurons = config["lstm_neurons"] 
        epochs = config["epochs"]
        batch_size = config["batch_size"]

        dataset_dirname = f"ws{window_size}_ss{step_size}_et{expected_timesteps}_lstm{lstm_neurons}_{epochs}epochs_bs{batch_size}"
        data_save_dir = os.path.join(save_dir, dataset_dirname, 'test_data')
        os.makedirs(data_save_dir, exist_ok=True)

        for label, path in paths.items():
            file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
            
            for batch_idx, file_batch in enumerate(np.array_split(file_paths, np.ceil(len(file_paths) / files_to_append))):
                concatenated_audio = np.concatenate([load_audio_file(fp, sample_rate) for fp in file_batch if load_audio_file(fp, sample_rate) is not None], axis=0)
                windows = sliding_window(concatenated_audio, window_size, step_size, sample_rate)

                for i in range(0, len(windows) - expected_timesteps + 1, expected_timesteps):
                    sequence = windows[i:i + expected_timesteps]
                    if len(sequence) == expected_timesteps:
                        features = np.array([extract_features(w, sample_rate) for w in sequence])
                        sequence_metadata = {
                            "file_paths": [os.path.basename(fp) for fp in file_batch],
                            "sequence_start": i,
                            "sequence_end": i + expected_timesteps
                        }
                        # Constructing the npz file name with additional metadata information
                        save_path = os.path.join(data_save_dir, f"{label}_config{config['window_size']}_batch{batch_idx}_seq{i}.npz")
                        # Save both features and metadata
                        np.savez_compressed(save_path, features=features, metadata=sequence_metadata)

def main(evaluation_directory):
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    paths = {
        'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset'
        # 'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_set',
        # 'validation':'/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_validation_set'
    }
    # concatenate_and_process_files(paths, SAMPLE_RATE, hyperparameters_combinations, files_to_append=2, save_dir=evaluation_directory)
    save_features_with_metadata(paths, SAMPLE_RATE, hyperparameters_combinations, 30, evaluation_directory)


if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v2'
    main(evaluation_directory)