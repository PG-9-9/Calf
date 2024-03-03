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

# Utility Functions:
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

def parse_filename_datetime(filename):
    """Parse the date and time from a given filename."""
    parts = filename.split('_')
    date_str = parts[1]
    time_str = parts[2].split('.')[0]  # remove the file extension
    datetime_obj = datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H-%M-%S')
    return datetime_obj

def determine_aggregation_level(file_mapping):
    """Determine whether to aggregate MSE by days or hours."""
    dates = set()
    for filename in file_mapping:
        datetime_obj = parse_filename_datetime(filename)
        dates.add(datetime_obj.date())
    
    if len(dates) >= 2:
        return 'day'
    else:
        return 'hour'
    
def count_anomalies(mse_test, thresholds):
    anomalies_count = [np.sum(mse_test > threshold) for threshold in thresholds]
    return anomalies_count

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
    overlap_audio = np.array([])
    
    for label, path in paths.items():
        file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            windows, temp_labels, overlap_audio, file_origins = process_batch_with_tracking(batch_files, label, sample_rate, window_size, step_size, overlap_audio)
            # Yield windows, labels, and origins for each batch
            yield np.array(windows), np.array(temp_labels), file_origins

def process_batch_with_tracking(file_paths, label, sample_rate, window_size, step_size, overlap_audio):
    batch_audio = overlap_audio
    file_origins = []  # Track the origin file for each window
    for file_path in file_paths:
        try:
            audio, _ = librosa.load(file_path, sr=sample_rate)
            batch_audio = np.concatenate((batch_audio, audio))
            # Assuming all windows from this file have the same label
            file_origins.extend([os.path.basename(file_path)] * int(len(audio) / (window_size * sample_rate)))
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
            continue
    windows, overlap_audio = sliding_window(batch_audio, window_size, step_size, sample_rate)
    labels = [label] * len(windows)
    # Adjust the length of file_origins to match the number of windows
    file_origins = file_origins[:len(windows)]
    return windows, labels, overlap_audio, file_origins

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

def model_evaluation(autoencoder, X_test, file_mapping, evaluation_directory, model_type):
    reconstructed_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=(1, 2))

    # Determine anomaly threshold and identify anomalies
    threshold = np.percentile(mse_test, 95)
    anomalies = mse_test > threshold
    logging.info(f"Detected {np.sum(anomalies)} anomalies out of {len(X_test)} windows, based on threshold {threshold}.")
    
    if file_mapping:
        # Aggregation of MSE by time (day or hour)
        aggregation_level = determine_aggregation_level(file_mapping)
        aggregated_mse = {}

        for i, mse in enumerate(mse_test):
            datetime_obj = parse_filename_datetime(file_mapping[i])
            if aggregation_level == 'day':
                key = datetime_obj.strftime('%Y-%m-%d')
            else:  # aggregation_level == 'hour'
                key = datetime_obj.strftime('%Y-%m-%d-h%H')

            if key in aggregated_mse:
                aggregated_mse[key].append(mse)
            else:
                aggregated_mse[key] = [mse]

        # Calculate average MSE for each aggregation key
        avg_mse = {k: np.mean(v) for k, v in aggregated_mse.items()}

        # Plotting average MSE by day or hour
        plt.figure(figsize=(15, 6))
        plt.bar(avg_mse.keys(), avg_mse.values(), color='red')
        plt.xticks(rotation=90)
        plt.xlabel('Aggregation Key')
        plt.ylabel('Average MSE')
        title = 'Average MSE per ' + ('Day' if aggregation_level == 'day' else 'Hour')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_directory, f'{title.lower().replace(" ", "_")}_{model_type}.png'))
        plt.close()
    else:
        print("File mapping not provided or not applicable.")
        
    # Utilize file_mapping for detailed anomaly analysis
    anomaly_file_mappings = [file_mapping[i] for i in range(len(file_mapping)) if anomalies[i]]
    
    # Example of logging detailed anomaly information
    unique_anomalous_files = set(anomaly_file_mappings)
    logging.info(f"Unique files with anomalies: {unique_anomalous_files}")

    # Visualizations and saving results (MSE distribution as an example)
    plt.figure(figsize=(10, 6))
    plt.hist(mse_test, bins=50, color='blue', alpha=0.7, label='Test MSE')
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of MSE on Test Data")
    plt.legend()
    plt.savefig(os.path.join(evaluation_directory, f'mse_distribution_{model_type}.png'))
    plt.close()

    # Save the model
    model_save_path = os.path.join(evaluation_directory, f'{model_type}_model.h5')
    try:
        autoencoder.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Failed to save the model. Error: {str(e)}")
        
        
def hyperparameter_tuning(root_path, evaluation_path, normal_data_path, abnormal_data_path, config_list, use_lstm=True):
    global SAMPLE_RATE
    for config in config_list:
        window_size, step_size, lstm_neurons, epochs, batch_size = config

        # Model directory for saving models and evaluations
        model_directory = create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size)

        # Accumulate features and labels for all batches
        accumulated_features, accumulated_labels = [], []

        # Process normal data for training in batches
        normal_paths = {'normal': normal_data_path}
        for windows, labels, _ in batch_process_audio_files(normal_paths, SAMPLE_RATE, window_size, step_size, batch_size):
            features = extract_features(windows, SAMPLE_RATE)
            accumulated_features.extend(features)
            accumulated_labels.extend(labels)

        # Prepare data for model input
        X_train_reshaped, _, y_train, _ = prepare_data(np.array(accumulated_features), np.array(accumulated_labels))

        # Train the model
        autoencoder = simplified_autoencoder_with_lstm(1, X_train_reshaped.shape[-1], lstm_neurons)
        autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

        # Reset for abnormal data processing
        accumulated_features, accumulated_labels, all_file_mappings = [], [], []

        # Process abnormal data for testing in batches
        abnormal_paths = {'abnormal': abnormal_data_path}
        for test_windows, test_labels, file_mapping in batch_process_audio_files(abnormal_paths, SAMPLE_RATE, window_size, step_size, batch_size):
            test_features = extract_features(test_windows, SAMPLE_RATE)
            accumulated_features.extend(test_features)
            accumulated_labels.extend(test_labels)
            all_file_mappings.extend(file_mapping)

        # Prepare test data for model evaluation
        X_test_reshaped, _, _, _ = prepare_data(np.array(accumulated_features), np.array(accumulated_labels))

        # Evaluate the model using accumulated abnormal data and file mappings
        model_evaluation(autoencoder, X_test_reshaped, all_file_mappings, evaluation_directory=model_directory, model_type='lstm_autoencoder')                

if __name__ == "__main__":
    try:
        root_path = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE"
        normal_data_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset"
        abnormal_data_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset"
        storage_path='/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3'
        print(f'current_directory : {os.getcwd()}')
        if os.path.exists(normal_data_path):
            print("It existssssss")
        config_list = [(10, 5, 32, 50, 32), (15, 7, 64, 100, 64) ]        # Hyperparameter Configurations [Window, Steps, LSTM , Epochs, Batch Size(for the audio files.)]
        hyperparameter_tuning(root_path, storage_path, normal_data_path, abnormal_data_path, config_list, use_lstm=True)

    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)