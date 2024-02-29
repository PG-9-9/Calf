import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import tensorflow as tf
import librosa
import os
import psutil
import time
from multiprocessing import Pool

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, auc
)

from tensorflow.keras import layers, models, Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Input, BatchNormalization, LSTM, RepeatVector,
    TimeDistributed
)
from datetime import datetime

from tensorflow.keras.callbacks import EarlyStopping
import logging

# Initialize logging
SAMPLE_RATE = 44100  # sample rate constant
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
current_datetime = datetime.now()  
logging.info(f"AutoEncoder last ran on: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Feature Extraction:

# MFCCs 
def extract_mfccs(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Spectral Features 
def extract_spectral_features(audio, sample_rate):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

# Temporal Features
def extract_temporal_features(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

#  Additional features
def extract_additional_features(audio, sample_rate):
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    
    return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)

# Combining all the features 
def extract_features(audio_windows, sample_rate):
    features = []
    for window in audio_windows:
        mfccs = extract_mfccs(window, sample_rate)
        spectral_features = extract_spectral_features(window, sample_rate)
        temporal_features = extract_temporal_features(window)
        additional_features = extract_additional_features(window, sample_rate)
        all_features = np.concatenate([mfccs, spectral_features, temporal_features, additional_features])
        features.append(all_features)
    return np.array(features)

# Audio Processing and Windowing for Batch Processing.
def batch_process_audio_files(paths, sample_rate, window_size, step_size, batch_size):
    all_windows = []
    all_labels = []
    overlap_audio = np.array([])

    for label, path in paths.items():
        print(f"Processing path for label '{label}': {path}")  # Debug print to check the path
        if not os.path.exists(path):
            logging.error(f"Directory does not exist: {path}")
            continue  # Skip this iteration if path doesn't exist

        file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            windows, temp_labels, overlap_audio = process_batch(batch_files, label, sample_rate, window_size, step_size, overlap_audio)
            all_windows.extend(windows)
            all_labels.extend(temp_labels)

    return np.array(all_windows), np.array(all_labels)

def process_batch(file_paths, label, sample_rate, window_size, step_size, overlap_audio):
    batch_audio = overlap_audio
    for file_path in file_paths:
        try:
            audio, _ = librosa.load(file_path, sr=sample_rate)
            batch_audio = np.concatenate((batch_audio, audio))
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
            continue
    windows, overlap_audio = sliding_window(batch_audio, window_size, step_size, sample_rate)
    labels = [label] * len(windows)
    return windows, labels, overlap_audio

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []
    start = 0
    while start + num_samples_per_window <= len(audio):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
        start += step_samples
    overlap_audio = audio[start:]
    return windows, overlap_audio

# Data Processing for Model input.
def prepare_data(features, labels):
    # Adjusted to split data while preserving temporal order
    split_index = int(len(features) * 0.8)
    X_train, X_val = features[:split_index], features[split_index:]
    y_train, y_val = labels[:split_index], labels[split_index:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    timesteps = 1  
    n_features = X_train_scaled.shape[1]
    X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
    X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))
    
    return X_train_reshaped, X_val_reshaped, y_train, y_val

# Model Buliding LSTM AutoEncoder
def simplified_autoencoder_with_lstm(timesteps, n_features, lstm_neurons):
    input_layer = Input(shape=(timesteps, n_features))

    # Encoder
    encoder = LSTM(lstm_neurons, activation='relu', return_sequences=False)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)

    # Repeat Vector to turn output into timesteps again
    repeat_vector = RepeatVector(timesteps)(encoder)

    # Decoder
    decoder = LSTM(lstm_neurons, activation='relu', return_sequences=True)(repeat_vector)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.1)(decoder)
    output_layer = TimeDistributed(Dense(n_features))(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def model_evaluation(autoencoder, X_test, evaluation_directory, model_type):
    # Generate predictions for the test set
    reconstructed_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=(1, 2))
    
    # Plot MSE distribution for the test set
    plt.figure(figsize=(10, 6))
    plt.hist(mse_test, bins=50, color='blue', alpha=0.7, label='Test MSE')
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of MSE on Abnormal Test Data")
    plt.legend()
    plt.savefig(os.path.join(evaluation_directory, f'mse_distribution_{model_type}.png'))
    plt.close()

    # An anomaly threshold (95th percentile of MSE)
    threshold = np.percentile(mse_test, 95)
    
    # Identify anomalies (where MSE exceeds the threshold)
    anomalies = mse_test > threshold
    num_anomalies = np.sum(anomalies)
    logging.info(f"Detected {num_anomalies} anomalies out of {len(X_test)} windows, based on threshold {threshold}")

    # Save the model
    model_save_path = os.path.join(evaluation_directory, f'{model_type}_model.h5')
    try:
        autoencoder.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Failed to save the model. Error: {str(e)}")

# Utility Functions
def create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size):
    model_directory = os.path.join(root_path, f"model_ws{window_size}_ss{step_size}_ln{lstm_neurons}_e{epochs}_bs{batch_size}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    return model_directory

def parse_date_time_from_filename(filename):
    parts = filename.split('_')
    date_part = parts[1]
    time_part = parts[2].split('.')[0]
    date_time_obj = datetime.strptime(f'{date_part} {time_part}', '%Y-%m-%d %H-%M-%S')
    return date_time_obj

def hyperparameter_tuning(root_path, evaluation_path, normal_data_path, abnormal_data_path, config_list, use_lstm=True):
    global SAMPLE_RATE
    for config in config_list:
        window_size, step_size, lstm_neurons, epochs, batch_size = config

        # Processing normal audio for training
        paths = {
            'normal': normal_data_path,
            'abnormal': abnormal_data_path
        } 
         
        windows, labels = batch_process_audio_files(paths, SAMPLE_RATE, window_size, step_size, batch_size)
        features = extract_features(windows, SAMPLE_RATE)
        X_train_reshaped, X_val_reshaped, y_train, y_val = prepare_data(features, labels)
        
        # Model training
        model_directory = create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size)
        if use_lstm:
            autoencoder = simplified_autoencoder_with_lstm(X_train_reshaped.shape[1], X_train_reshaped.shape[2], lstm_neurons)
            autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=epochs, batch_size=batch_size, validation_data=(X_val_reshaped, X_val_reshaped), callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=1)
        
        # Processing abnormal audio for testing
        test_windows, _ = batch_process_audio_files(paths, SAMPLE_RATE, window_size, step_size, batch_size)
        test_features = extract_features(test_windows, SAMPLE_RATE)
        X_test_reshaped, _, _, _ = prepare_data(test_features, np.zeros(len(test_features)))  # Labels not used for anomaly detection
        
        # Model evaluation on abnormal audio
        model_evaluation(autoencoder, X_test_reshaped, evaluation_path, 'lstm')  
        
if __name__ == "__main__":
    try:
        root_path = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE"
        normal_data_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset"
        abnormal_data_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset"
        storage_path='/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files'
        print(f'current_directory : {os.getcwd()}')
        if os.path.exists(normal_data_path):
            print("It existssssss")
        config_list = [(10, 5, 32, 50, 32), (15, 7, 64, 100, 64) ]        # Hyperparameter Configurations [Window, Steps, LSTM , Epochs, Batch Size(for the audio files.)]
        hyperparameter_tuning(root_path, storage_path, normal_data_path, abnormal_data_path, config_list, use_lstm=True)

    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)