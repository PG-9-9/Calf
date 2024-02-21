import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import tensorflow as tf
import librosa
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)

from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Input, BatchNormalization, LSTM, RepeatVector,
    TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Initialize logging

current_datetime=datetime.datetime.now()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print(f"AutoEncoder last ran on: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Sliding Window

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []
    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
    return windows

# Feature Extraction:

# MFCCs (Power Spectrum)
def extract_mfccs(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Spectral Features (spectral centroid, spectral roll-off, and spectral contrast):
def extract_spectral_features(audio, sample_rate):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

# Temporal Features ( zero-crossing rate and autocorrelation):
def extract_temporal_features(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

# Load audio files and apply sliding windows

def load_and_window_audio_files(path, label, window_size, step_size, sample_rate):
    audio_windows = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            file_path = os.path.join(path, filename)
            audio, _ = librosa.load(file_path, sr=sample_rate)
            windows = sliding_window(audio, window_size, step_size, sample_rate)
            audio_windows.extend(windows)
            labels.extend([label] * len(windows))
    return audio_windows, labels

# Feature extraction for each window

def extract_features(audio_windows, sample_rate):
    features = []
    for window in audio_windows:
        mfccs = extract_mfccs(window, sample_rate)
        spectral_features = extract_spectral_features(window, sample_rate)
        temporal_features = extract_temporal_features(window)
        all_features = np.concatenate([mfccs, spectral_features, temporal_features])
        features.append(all_features)
    return np.array(features)

# Simplified LSTM autoencoder

def enhanced_autoencoder_with_lstm(timesteps, n_features):
    input_layer = Input(shape=(timesteps, n_features))

    # Very simple Encoder with LSTM
    encoder = LSTM(16, activation='relu', return_sequences=False)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)

    # Repeat Vector
    repeat_vector = RepeatVector(timesteps)(encoder)

    # Very simple Decoder with LSTM
    decoder = LSTM(16, activation='relu', return_sequences=True)(repeat_vector)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.1)(decoder)
    output_layer = TimeDistributed(Dense(n_features))(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def process_data(features):
    logging.info("Starting data processing")
    X_train, X_val = train_test_split(features, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    logging.info("Completed scaling")

    # Reshape data for LSTM
    timesteps = 1  # Assuming each window is treated as a separate sequence
    n_features = X_train_scaled.shape[1]
    X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
    X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))
    logging.info(f"Data reshaped: {X_train_reshaped.shape}")

    return X_train_reshaped, X_val_reshaped

def train_model(X_train, X_val):
    logging.info(f"Model training with data shape: {X_train.shape}")
    try:
        autoencoder = enhanced_autoencoder_with_lstm(X_train.shape[1], X_train.shape[2])
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=1)
        logging.info("Model training completed")
    except Exception as e:
        logging.error("An error occurred during model training", exc_info=True)

if __name__ == "__main__":
    try:
        normal_calf_path = "Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset"
        abnormal_calf_path = "Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset"

        window_size = 10  # in seconds
        step_size = 5  # in seconds
        sample_rate = 44100  

        # Load and window the data
        normal_audio_windows, normal_labels = load_and_window_audio_files(normal_calf_path, label=0, window_size=window_size, step_size=step_size, sample_rate=sample_rate)
        abnormal_audio_windows, abnormal_labels = load_and_window_audio_files(abnormal_calf_path, label=1, window_size=window_size, step_size=step_size, sample_rate=sample_rate)

        # Extract features for windows
        normal_features = extract_features(normal_audio_windows, sample_rate)
        abnormal_features = extract_features(abnormal_audio_windows, sample_rate)

        normal_features = np.array(normal_features)
        X_train_reshaped, X_val_reshaped = process_data(normal_features)
        train_model(X_train_reshaped, X_val_reshaped)

        # Test with synthetic data
        synthetic_data = np.random.rand(30, 1, 18)
        train_model(synthetic_data, synthetic_data)

    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)