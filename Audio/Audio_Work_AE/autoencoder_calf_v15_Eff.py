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
SAMPLE_RATE = 44100  # Sample rate constant
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
current_datetime = datetime.datetime.now()
logging.info(f"AutoEncoder last ran on: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Audio Processing 

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []
    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
    return windows

def extract_mfccs(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def extract_spectral_features(audio, sample_rate):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

def extract_temporal_features(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

def extract_additional_features(audio, sample_rate):
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)


# Data Processing

def load_and_window_audio_files_with_tracking(path, window_size, step_size, sample_rate, file_limit=100):
    audio_windows, filenames = [], []
    file_count = 0
    for filename in os.listdir(path):
        if filename.endswith('.wav') and file_count < file_limit:
            file_path = os.path.join(path, filename)
            audio, _ = librosa.load(file_path, sr=sample_rate)
            windows = sliding_window(audio, window_size, step_size, sample_rate)
            audio_windows.extend(windows)
            # Each window in a file shares the same filename, useful for later analysis
            filenames.extend([filename] * len(windows))
            file_count += 1
    return np.array(audio_windows), filenames

def extract_features(audio_windows, sample_rate):
    features = []
    for window in audio_windows:
        mfccs = extract_mfccs(window, sample_rate)
        spectral_features = extract_spectral_features(window, sample_rate)
        temporal_features = extract_temporal_features(window)
        additional_features = extract_additional_features(window, sample_rate)
        all_features = np.hstack([mfccs, *spectral_features, *temporal_features, *additional_features])
        features.append(all_features)
    return np.array(features)

# Model Preparation and Evaluation

def create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size):
    model_directory = os.path.join(root_path, f"model_ws{window_size}_ss{step_size}_ln{lstm_neurons}_e{epochs}_bs{batch_size}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    return model_directory

def prepare_data(features, labels, use_lstm):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if use_lstm:
        timesteps = 1
        n_features = X_train_scaled.shape[1]
        X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
        X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))
    else:
        X_train_reshaped = X_train_scaled
        X_val_reshaped = X_val_scaled
    return X_train_reshaped, X_val_reshaped, y_train, y_val

def model_evaluation(autoencoder, X_test, y_test, evaluation_directory, model_type, X_abnormal, filenames_abnormal):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Original evaluation on test set
    reconstructed_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=(1, 2))
    
    # Original evaluation on abnormal set
    reconstructed_abnormal = autoencoder.predict(X_abnormal)
    mse_abnormal = np.mean(np.power(X_abnormal - reconstructed_abnormal, 2), axis=(1, 2))
    
    # Precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, mse_test)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
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
    plt.xlabel('Sample Index')
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
    
    autoencoder.save(os.path.join(evaluation_directory, 'model.keras'))
    logging.info(f"Model Evaluation Metrics - Optimal Threshold: {optimal_threshold}, Accuracy: {accuracy_score(y_test, optimal_predictions)}, Precision: {precision_score(y_test, optimal_predictions)}, Recall: {recall_score(y_test, optimal_predictions)}, F1 Score: {f1_score(y_test, optimal_predictions)}")

    # Aggregated MSE evaluation
    # Predict and calculate MSE for each window in abnormal set
    aggregated_mse = {}
    for mse, filename in zip(mse_abnormal, filenames_abnormal):
        if filename not in aggregated_mse:
            aggregated_mse[filename] = []
        aggregated_mse[filename].append(mse)
    
    # Calculate mean MSE per file
    mean_mse_per_file = {filename: np.mean(mse) for filename, mse in aggregated_mse.items()}

    # Plotting aggregated MSE per file
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(mean_mse_per_file)), list(mean_mse_per_file.values()), align='center')
    plt.xticks(range(len(mean_mse_per_file)), list(mean_mse_per_file.keys()), rotation=90)
    plt.title('Aggregated MSE per File')
    plt.xlabel('File')
    plt.ylabel('Mean MSE')
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_directory, f'aggregated_mse_{model_type}_{timestamp}.png'))
    plt.close()

def simplified_autoencoder_with_lstm(timesteps, n_features, lstm_neurons):
    input_layer = Input(shape=(timesteps, n_features))
    encoder = LSTM(lstm_neurons, activation='relu', return_sequences=False)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)
    repeat_vector = RepeatVector(timesteps)(encoder)
    decoder = LSTM(lstm_neurons, activation='relu', return_sequences=True)(repeat_vector)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.1)(decoder)
    output_layer = TimeDistributed(Dense(n_features))(decoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# Experiment logging and model training.
    
def hyperparameter_tuning(root_path, evaluation_path, data_path, config_list, use_lstm=True):
    global SAMPLE_RATE
    for config in config_list:
        window_size, step_size, lstm_neurons, epochs, batch_size = config
        logging.info(f"Config - Window: {window_size}s, Step: {step_size}s, LSTM Neurons: {lstm_neurons}, Epochs: {epochs}, Batch Size: {batch_size}")
        audio_windows, filenames = load_and_window_audio_files_with_tracking(data_path['normal'], window_size, step_size, SAMPLE_RATE, file_limit=100)  # Adjusted file_limit for scalability
        features = extract_features(audio_windows, SAMPLE_RATE)
        X_train_reshaped, X_val_reshaped, y_train, y_val = prepare_data(features, np.zeros(len(features)), use_lstm)
        model_directory = create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size)
        
        autoencoder = simplified_autoencoder_with_lstm(X_train_reshaped.shape[1], X_train_reshaped.shape[2], lstm_neurons)
        autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=epochs, batch_size=batch_size, validation_data=(X_val_reshaped, X_val_reshaped), callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=1)
        
        # Load and preprocess abnormal dataset for evaluation
        X_abnormal, filenames_abnormal = load_and_window_audio_files_with_tracking(data_path['abnormal'], window_size, step_size, SAMPLE_RATE, file_limit=100)  # Assuming abnormal dataset processing
        X_abnormal_features = extract_features(X_abnormal, SAMPLE_RATE)
        X_abnormal_reshaped = X_abnormal_features.reshape((-1, 1, X_abnormal_features.shape[1]))  # Reshape for LSTM input
        
        # Evaluate model with both normal (validation set) and abnormal data
        model_evaluation(autoencoder, X_val_reshaped, y_val, model_directory, "LSTM_AE", X_abnormal_reshaped, filenames_abnormal)

if __name__ == "__main__":
    try:
        data_path = {
            'normal': "Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset",
            'abnormal': "Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset"
        }
        config_list = [
            (10, 5, 32, 50, 32),
            (15, 7, 64, 100, 64)
        ]
        root_path = "Calf_Detection/Audio/Audio_Work_AE"
        evaluation_path = 'Calf_Detection/Audio/Audio_Work_AE/New_Files'
        hyperparameter_tuning(root_path, evaluation_path, data_path, config_list, use_lstm=True)
        
    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)
