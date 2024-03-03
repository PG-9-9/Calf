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
SAMPLE_RATE = 44100  # rate constant (Pre-defined)
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
    time_str = parts[2].split('.')[0]  # remove the file extension(.wav)
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
def extract_features(audio, sample_rate):
    mfccs = extract_mfccs(audio, sample_rate)
    spectral_centroids, spectral_rolloff, spectral_contrast = extract_spectral_features(audio, sample_rate)
    zero_crossing_rate, autocorrelation = extract_temporal_features(audio)
    chroma_stft, spec_bw, spec_flatness, rolloff, rms = extract_additional_features(audio, sample_rate)
    features = np.concatenate([mfccs, [spectral_centroids, spectral_rolloff, spectral_contrast, zero_crossing_rate, autocorrelation, chroma_stft, spec_bw, spec_flatness, rolloff, rms]])
    return features

# Data Processing and sliding window
# def sliding_window(audio, window_size, step_size, sample_rate):
#     num_samples_per_window = int(window_size * sample_rate)
#     step_samples = int(step_size * sample_rate)
#     num_windows = (len(audio) - num_samples_per_window) // step_samples + 1
#     windows = np.array([audio[i * step_samples: i * step_samples + num_samples_per_window] for i in range(num_windows)])
#     return windows

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    num_windows = (len(audio) - num_samples_per_window) // step_samples + 1
    return num_windows

def preprocess_and_extract_features(file_path, sample_rate=SAMPLE_RATE, window_size=0.5, step_size=0.25):
    try:
        audio, _ = librosa.load(file_path.numpy().decode(), sr=sample_rate)
        windows = sliding_window(audio, window_size, step_size, sample_rate)
        features = np.vstack([extract_features(window, sample_rate) for window in windows])
    except Exception as e:
        logging.info(f"Processed {file_path.numpy().decode()}: Features shape {features.shape}")  # Return an empty array of shape (0, feature_length) if error occurs
        features = np.empty((0, 29))  
    return features.astype(np.float32)

def audio_to_features(file_path, sample_rate=tf.constant(SAMPLE_RATE, dtype=tf.float32), window_size=tf.constant(0.5, dtype=tf.float32), step_size=tf.constant(0.25, dtype=tf.float32)):
    features = tf.py_function(func=preprocess_and_extract_features, inp=[file_path, sample_rate, window_size, step_size], Tout=tf.float32)
    # Dynamically setting the shape to None for variable-length sequences of features
    features.set_shape([None, 23])  # Adjust the second None to the actual number of features if known
    return features

def create_tf_dataset(file_paths, batch_size, sample_rate, window_size, step_size):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(lambda x: audio_to_features(x, sample_rate, window_size, step_size), num_parallel_calls=tf.data.AUTOTUNE)
    # Correctly handle datasets where each item can have a variable number of windows
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_max_seq_length_from_files(file_paths, sample_rate=44100, window_size=0.5, step_size=0.25):
    max_length = 0
    for file_path in file_paths:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        windows = sliding_window(audio, window_size, step_size, sample_rate)
        if len(windows) > max_length:
            max_length = len(windows)
    return max_length

def simplified_autoencoder_with_lstm(feature_dim, lstm_neurons):
    input_layer = Input(shape=(None, feature_dim))  # None allows for variable sequence length
    # Encoder
    encoded = LSTM(lstm_neurons, activation='relu', return_sequences=True)(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = LSTM(lstm_neurons, activation='relu')(encoded)  # Last LSTM does not return sequences

    # Decoder
    repeated = RepeatVector(tf.shape(input_layer)[1])(encoded)  # Dynamically repeat based on input length
    decoded = LSTM(lstm_neurons, activation='relu', return_sequences=True)(repeated)
    decoded = BatchNormalization()(decoded)
    decoded = LSTM(lstm_neurons, activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(feature_dim))(decoded)  # Predict the feature vector for each time step

    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def model_evaluation(autoencoder, X_test, file_mapping, evaluation_directory, model_type):
    
    reconstructed_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=(1, 2))
    
    min_threshold = min(mse_test)  # minimum MSE value
    max_threshold = max(mse_test)  # maximum MSE value
    threshold_range = np.linspace(min_threshold, max_threshold, 10)  
    anomalies_count = count_anomalies(mse_test, threshold_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_range, anomalies_count, marker='o', linestyle='-')
    plt.xlabel('Threshold Value')
    plt.ylabel('Number of Anomalies Detected')
    plt.title('Number of Anomalies Detected vs. Threshold Value')
    plt.grid(True)
    plt.savefig(os.path.join(evaluation_directory, f'anomalies_vs_threshold_{model_type}.png'))
    plt.close()
    
    if file_mapping:
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
            
            # Plotting
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

def hyperparameter_tuning(root_path, evaluation_path, normal_data_path, abnormal_data_path, config_list):
    for window_size, step_size, lstm_neurons, epochs, batch_size in config_list:
        logging.info(f"Config: WS={window_size}, SS={step_size}, LSTM={lstm_neurons}, Epochs={epochs}, Batch={batch_size}")

        normal_file_paths = [os.path.join(normal_data_path, f) for f in os.listdir(normal_data_path) if f.endswith('.wav')]
        abnormal_file_paths = [os.path.join(abnormal_data_path, f) for f in os.listdir(abnormal_data_path) if f.endswith('.wav')]

        normal_dataset = create_tf_dataset(normal_file_paths, batch_size, SAMPLE_RATE, window_size, step_size)
        abnormal_dataset = create_tf_dataset(abnormal_file_paths, batch_size, SAMPLE_RATE, window_size, step_size)

        # Adjusting model creation to handle the dataset correctly
        feature_shape = 23 
        model = simplified_autoencoder_with_lstm(feature_shape, lstm_neurons)
        model.fit(normal_dataset, epochs=epochs, validation_data=abnormal_dataset, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
        model_evaluation(model, abnormal_dataset, evaluation_path, 'lstm_autoencoder')  
        
                  
if __name__ == "__main__":
    try:
        root_path = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE"
        normal_data_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_superset"
        abnormal_data_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_superset"
        storage_path='/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v2'
        # print(f'current_directory : {os.getcwd()}')
        # if os.path.exists(normal_data_path):
        #     print("It existssssss")
        # config_list = [(10, 5, 32, 50, 32), (15, 7, 64, 100, 64) ]        # Hyperparameter Configurations [Window, Steps, LSTM , Epochs, Batch Size(for the audio files.)]
        # hyperparameter_tuning(root_path, storage_path, normal_data_path, abnormal_data_path, config_list)
        all_file_paths = normal_data_path + abnormal_data_path  # Combine both normal and abnormal data paths
        for file_path in all_file_paths:
            print(f"Loading file: {file_path}")
            audio, _ = librosa.load(file_path, sr=44100)
        from os.path import isfile
        all_file_paths = [path for path in all_file_paths if isfile(path)]
        
        max_seq_length = get_max_seq_length_from_files(all_file_paths)
        
        
        print(f"Max sequence length: {max_seq_length}") 
    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)