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
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Initialize logging
SAMPLE_RATE = 44100  # sample rate constant
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
current_datetime = datetime.datetime.now()  
logging.info(f"AutoEncoder last ran on: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
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

#  Additional features
def extract_additional_features(audio, sample_rate):
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    
    return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)
# Load audio files and apply sliding windows

def load_and_window_audio_files(path, label, window_size, step_size, sample_rate, file_limit=26500):
    audio_windows = []
    labels = []
    file_count = 0
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            if file_count >= file_limit:
                break
            file_path = os.path.join(path, filename)
            audio, _ = librosa.load(file_path, sr=sample_rate)
            windows = sliding_window(audio, window_size, step_size, sample_rate)
            audio_windows.extend(windows)
            labels.extend([label] * len(windows))
            file_count += 1
    return audio_windows, labels

def process_data(features):
    logging.info("Starting data processing");
    X_train, X_val = train_test_split(features, test_size=0.2)#, random_state=42
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    logging.info("Completed scaling")

    # Reshape data for LSTM
    timesteps = 1  # Each window is treated as a separate sequence
    n_features = X_train_scaled.shape[1]
    X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
    X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))
    logging.info(f"Data reshaped: {X_train_reshaped.shape}")

    return X_train_reshaped, X_val_reshaped

def process_file(args):
    filename, path, window_size, step_size, sample_rate = args
    file_path = os.path.join(path, filename)
    audio, _ = librosa.load(file_path, sr=sample_rate)
    windows = sliding_window(audio, window_size, step_size, sample_rate)
    return windows

# Feature extraction for each window

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

# LSTM AutoEncoder
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

def model_evaluation(autoencoder, X_test, y_test, evaluation_directory, model_type, X_abnormal):
    # Generate a timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    
    # Predict on the test set
    reconstructed_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=(1, 2))
    
    # Predict on the entire abnormal dataset
    reconstructed_abnormal = autoencoder.predict(X_abnormal)
    mse_abnormal = np.mean(np.power(X_abnormal - reconstructed_abnormal, 2), axis=(1, 2))
    
    # Precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, mse_test)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)  # Handling NaN values 
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_predictions = (mse_test > optimal_threshold).astype(int)
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, optimal_predictions), annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Optimal Confusion Matrix')
    plt.savefig(os.path.join(evaluation_directory, f'confusion_matrix_{model_type}_{timestamp}.png'))
    plt.close()
    
    # MSE Error Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mse_abnormal, label='MSE Error')
    plt.title('MSE Error Over Test Set')
    plt.xlabel('Entire subset')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(os.path.join(evaluation_directory, f'mse_error_plot_{model_type}_{timestamp}.png'))
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, mse_test)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(evaluation_directory, f'roc_curve_{model_type}_{timestamp}.png'))
    plt.close()
    
    # Save the trained model
    autoencoder.save(os.path.join(evaluation_directory, 'model.keras'))

    # Log evaluation metrics
    logging.info(f"Model Evaluation Metrics - Optimal Threshold: {optimal_threshold}, Accuracy: {accuracy_score(y_test, optimal_predictions)}, Precision: {precision_score(y_test, optimal_predictions)}, Recall: {recall_score(y_test, optimal_predictions)}, F1 Score: {f1_score(y_test, optimal_predictions)}")

def create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size):
    model_directory = os.path.join(root_path, f"model_ws{window_size}_ss{step_size}_ln{lstm_neurons}_e{epochs}_bs{batch_size}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    return model_directory
 
def prepare_data(features, labels, use_lstm):
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Reshape data for LSTM input
    if use_lstm:
        timesteps = 1  
        n_features = X_train_scaled.shape[1]
        X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
        X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))
    else:
        X_train_reshaped = X_train_scaled
        X_val_reshaped = X_val_scaled

    return X_train_reshaped, X_val_reshaped, y_train, y_val

def hyperparameter_tuning(root_path, evaluation_path, data_path, config_list, use_lstm=True):
    global SAMPLE_RATE  # SAMPLE_RATE defined globally
    for config in config_list:
        window_size, step_size, lstm_neurons, epochs, batch_size = config
        logging.info(f"Config - Window: {window_size}s, Step: {step_size}s, LSTM Neurons: {lstm_neurons}, Epochs: {epochs}, Batch Size: {batch_size}")

        # Initialize lists to collect features and labels
        all_features = []
        all_labels = []

        # Process normal audio files
        normal_audio_windows, _ = load_and_window_audio_files(data_path['normal'], 0, window_size, step_size, SAMPLE_RATE)
        normal_features = extract_features(normal_audio_windows, SAMPLE_RATE)
        all_features.extend(normal_features)
        all_labels.extend([0] * len(normal_features))

        # Process abnormal audio files
        abnormal_audio_windows, _ = load_and_window_audio_files(data_path['abnormal'], 1, window_size, step_size, SAMPLE_RATE)
        abnormal_features = extract_features(abnormal_audio_windows, SAMPLE_RATE)
        all_features.extend(abnormal_features)
        all_labels.extend([1] * len(abnormal_features))

        # Convert lists to numpy arrays for processing
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        # Prepare data: split, scale, and reshape
        X_train_reshaped, X_val_reshaped, y_train, y_val = prepare_data(all_features, all_labels, use_lstm)
        
        # Prepare entire abnormal dataset for MSE error plotting
        X_abnormal_reshaped, _, _, _ = prepare_data(abnormal_features, np.ones(len(abnormal_features)), use_lstm)

        # Create model directory based on configuration
        model_directory = create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size)

        # Select the model based on LSTM flag
        if use_lstm:
            autoencoder = simplified_autoencoder_with_lstm(X_train_reshaped.shape[1], X_train_reshaped.shape[2], lstm_neurons)
        
        # Train the model
        autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val_reshaped, X_val_reshaped),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=1)

        # Evaluate the model and save evaluation plots and the model itself
        model_type_label = 'lstm' if use_lstm else 'non_lstm'
        model_evaluation(autoencoder, X_val_reshaped, y_val, evaluation_path, model_type_label,X_abnormal_reshaped)


def train_model(X_train, X_val):
    logging.info(f"Model training with data shape: {X_train.shape}")
    try:
        autoencoder = simplified_autoencoder_with_lstm(X_train.shape[1], X_train.shape[2])
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=1)
        logging.info("Model training completed")
    except Exception as e:
        logging.error("An error occurred during model training", exc_info=True)

if __name__ == "__main__":
    try:
        data_path = {
            'normal': "Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset",
            'abnormal': "Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset"
        }
        sample_rate = 44100  # Sample rate of audio files (Pre-determined)

        # Hyperparameter Configurations
        # Window, Steps, LSTM , Epochs, Batch Size
        config_list = [
            (10, 5, 32, 50, 32), 
            # (10, 5, 32, 100, 32),    
            (15, 7, 64, 100, 64) # Currently this is the best.
            # (15, 7, 64, 200, 64)
        ]
        

        root_path = "Calf_Detection/Audio/Audio_Work_AE"
        evaluation_path='Calf_Detection/Audio/Audio_Work_AE/New_Files'
        # Hyperparameter tuning for LSTM Autoencoder
        hyperparameter_tuning(root_path, evaluation_path, data_path, config_list, use_lstm=True)

        # Hyperparameter tuning for Dense Autoencoder
        # hyperparameter_tuning(root_path, evaluation_path , data_path, config_list, use_lstm=False)

    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)