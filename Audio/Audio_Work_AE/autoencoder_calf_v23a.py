import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import librosa
import os
import logging
import time
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.model_selection import train_test_split
from logging import NullHandler
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LeakyReLU, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, load_model

# Constants
SAMPLE_RATE = 44100
TOTAL_FEATURES = 33
LOGGING_ENABLED = False
# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class ResetStatesCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()

# Setup a global logging directory
log_dir = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()  # Clear existing handlers

    if LOGGING_ENABLED:
        handler = logging.FileHandler(log_file)        
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    else:
        # Add a NullHandler when logging is disabled
        logger.addHandler(NullHandler())
    
    logger.setLevel(level)
    return logger

feature_extraction_logger = setup_logger('feature_extraction', os.path.join(log_dir, 'feature_extraction.log'))

def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"

# 5 * 25 = 125,   
hyperparameters_combinations = [
    {"window_size": 5, "step_size": 2.5, "expected_timesteps": 23, "lstm_neurons": 128, "epochs": 20, "batch_size": 30}
    # {"window_size": 10, "step_size": 5, "expected_timesteps": 11, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
    # {"window_size": 15, "step_size": 7.5, "expected_timesteps": 7, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
]

def create_model_directory(root_path, config):
    model_dir = os.path.join(root_path, "model_{}".format("_".join(map(str, config))))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def load_audio_file(file_path, sample_rate=SAMPLE_RATE, logger=None):
    start_time = time.time()
    audio, _ = librosa.load(file_path, sr=sample_rate)
    if logger:
        logger.info(f"Loaded audio file {file_path} in {convert_seconds(time.time() - start_time)}.")
    return audio

def extract_spectral_features(audio, sample_rate, logger=None):
    start_time = time.time()
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    if logger:
        logger.info(f"Spectral features extraction time: {convert_seconds(time.time() - start_time)}.")
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

def extract_temporal_features(audio, logger=None):
    start_time = time.time()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    if logger:
        logger.info(f"Temporal features extraction time: {convert_seconds(time.time() - start_time)}.")
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

def extract_additional_features(audio, sample_rate, logger=None):
    start_time = time.time()
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    if logger:
        logger.info(f"Additional features extraction time: {convert_seconds(time.time() - start_time)}.")
    return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)

def extract_raw_audio_features(audio, num_samples, logger=None):
    start_time = time.time()
    step = len(audio) // num_samples
    raw_features = audio[::step][:num_samples]
    if logger:
        logger.info(f"Raw audio features extraction time: {convert_seconds(time.time() - start_time)}.")
    return raw_features

def extract_features(audio, sample_rate, feature_extraction_logger=None):
    # Aggregate all feature extraction processes
    start_time = time.time()
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)    # Extract MFCCs
    mfccs_processed = np.mean(mfccs.T,axis=0)
    spectral_features = extract_spectral_features(audio, sample_rate, feature_extraction_logger)
    temporal_features = extract_temporal_features(audio, feature_extraction_logger)
    additional_features = extract_additional_features(audio, sample_rate, feature_extraction_logger)
    raw_audio_features = extract_raw_audio_features(audio, 10, feature_extraction_logger)
    features = np.concatenate((mfccs_processed,spectral_features, temporal_features, additional_features, raw_audio_features))
    if features.shape[0] != TOTAL_FEATURES:
        raise ValueError(f"Feature extraction error: Expected {TOTAL_FEATURES} features, got {features.shape[0]}")
    if feature_extraction_logger:
        feature_extraction_logger.info(f"Total feature extraction time: {convert_seconds(time.time() - start_time)}.")
        feature_extraction_logger.info(f"Feature shape: {features.shape}, Total feature extraction time: {convert_seconds(time.time() - start_time)}.")

    return features

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []

    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
    return windows

def save_features_in_batches(paths, sample_rate, combination, output_dir, n_files_per_batch):
    window_size = combination["window_size"]
    step_size = combination["step_size"]
    expected_timesteps = combination["expected_timesteps"]
    batch_size = combination["batch_size"]

    feature_save_dir = os.path.join(output_dir, f"ws{window_size}_ss{step_size}_et{expected_timesteps}_bs{batch_size}")
    os.makedirs(feature_save_dir, exist_ok=True)
    
    batch_counter = 0
    sequence_features = []

    for label, path in paths.items():
        for file_path in sorted(os.listdir(path)):
            if file_path.endswith('.wav'):
                audio_file_path = os.path.join(path, file_path)
                audio, sr = librosa.load(audio_file_path, sr=sample_rate)

                # Divide audio into overlapping windows
                windows = sliding_window(audio, window_size, step_size, sample_rate)

                for window in windows:
                    # Extract features for each window
                    features = extract_features(window, sr)
                    sequence_features.append(features)

                    # Once we accumulate enough for a batch, save it
                    if len(sequence_features) == (batch_size * expected_timesteps):
                        batch_features = np.array(sequence_features).reshape(batch_size, expected_timesteps, -1)
                        np.savez_compressed(os.path.join(feature_save_dir, f'batch_{batch_counter}.npz'), features=batch_features)
                        sequence_features = []  # Reset for next batch
                        batch_counter += 1

    # Handle leftover features (last batch)
    if sequence_features:
        leftover_batch_size = len(sequence_features) // expected_timesteps
        if leftover_batch_size > 0:
            leftover_features = sequence_features[:leftover_batch_size * expected_timesteps]
            leftover_features = np.array(leftover_features).reshape(leftover_batch_size, expected_timesteps, -1)
            np.savez_compressed(os.path.join(feature_save_dir, f'batch_{batch_counter}_leftover.npz'), features=leftover_features)
                
def build_autoencoder(expected_timesteps, total_features, lstm_neurons):
    input_layer = Input(shape=(expected_timesteps, total_features))

    # Encoder
    x = LSTM(128, activation='relu', return_sequences=True)(input_layer)
    x = LSTM(64, activation='relu', return_sequences=False)(x)
    x = RepeatVector(expected_timesteps)(x)

    # Decoder
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    x = LSTM(128, activation='relu', return_sequences=True)(x)
    output_layer = TimeDistributed(Dense(total_features))(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

    
def create_dataset_from_npz(feature_dir, expected_timesteps, total_features, batch_size):
    def generator():
        for npz_file in sorted(os.listdir(feature_dir)):
            if npz_file.endswith('.npz'):
                data = np.load(os.path.join(feature_dir, npz_file))
                features = data['features']
                for feature in features:
                    if feature.shape[0] == expected_timesteps and feature.shape[1] == total_features:
                        yield feature, feature  # Correct shape
                    else:
                        # Log or handle features with incorrect shapes
                        print(f"Skipping feature with incorrect shape: {feature.shape}")

    output_types = (tf.float32, tf.float32)
    output_shapes = ((expected_timesteps, total_features), (expected_timesteps, total_features))

    dataset = tf.data.Dataset.from_generator(
        generator, 
        output_types=output_types, 
        output_shapes=output_shapes
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)  # Handle last smaller batch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def experiment_with_configurations(evaluation_directory, hyperparameters_combinations):
    for combination in hyperparameters_combinations:
        dataset_dirname = f"ws{combination['window_size']}_ss{combination['step_size']}_et{combination['expected_timesteps']}_bs{combination['batch_size']}"
        feature_dir = os.path.join(evaluation_directory, dataset_dirname)

        if not os.path.exists(feature_dir):
            logging.error(f"Feature directory does not exist: {feature_dir}")
            continue
        
        dataset = create_dataset_from_npz(feature_dir, combination['expected_timesteps'], TOTAL_FEATURES, combination['batch_size'])
        model = build_autoencoder(combination['expected_timesteps'], TOTAL_FEATURES, combination['lstm_neurons'])

        model.fit(dataset, epochs=combination['epochs'], callbacks=[ResetStatesCallback(), EarlyStopping(monitor='loss', patience=3)])

        model_save_path = os.path.join(evaluation_directory, dataset_dirname, "autoencoder_model.h5")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")


def main(evaluation_directory, enable_logging):
    global LOGGING_ENABLED
    LOGGING_ENABLED = enable_logging
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    paths = {
        'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset'
        # 'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_subset',
        # 'validation':'/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_validation_set'
    }
    for combination in hyperparameters_combinations:
        save_features_in_batches(paths, SAMPLE_RATE, combination, evaluation_directory, n_files_per_batch=30)
        print(f"Saved features in batches for combination: {combination}")   

    experiment_with_configurations(evaluation_directory, hyperparameters_combinations)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v4'
    main(evaluation_directory, enable_logging=False)