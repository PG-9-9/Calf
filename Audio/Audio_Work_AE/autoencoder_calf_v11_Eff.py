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
SAMPLE_RATE = 44100  # Define sample rate globally if it's constant
current_datetime=datetime.datetime.now()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def generate_audio_windows(path, window_size, step_size, sample_rate, batch_size=10):
    filenames = [f for f in os.listdir(path) if f.endswith('.wav')]
    pool = Pool(processes=os.cpu_count())

    for i in range(0, len(filenames), batch_size):
        batch_files = filenames[i:i + batch_size]
        windows_list = pool.map(process_file, [(filename, path, window_size, step_size, sample_rate) for filename in batch_files])
        
        for windows in windows_list:
            yield windows

    pool.close()
    pool.join()

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

# Non LSTM based Auto-encoder
def simplified_autoencoder(timesteps, n_features):
    input_layer = Input(shape=(n_features,))

    # Encoder
    encoder = Dense(128, activation='relu')(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(32, activation='relu')(encoder)

    # Decoder
    decoder = Dense(64, activation='relu')(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.1)(decoder)
    output_layer = Dense(n_features, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder


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

def model_evaluation(autoencoder, X_test, y_test, model_directory):
    reconstructed_test = autoencoder.predict(X_test)
    mse_test = np.mean(np.power(X_test - reconstructed_test, 2), axis=(1, 2))
    precisions, recalls, thresholds = precision_recall_curve(y_test, mse_test)
    f1_scores = np.where((precisions + recalls) == 0, 0, 2 * (precisions * recalls) / (precisions + recalls))    
    # Check for NaN values in F1 scores and handle them
    f1_scores = np.nan_to_num(f1_scores)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    optimal_predictions = (mse_test > optimal_threshold).astype(int)

    # Plot and save the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, optimal_predictions), annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Optimal Confusion Matrix')
    plt.savefig(os.path.join(model_directory, 'confusion_matrix.png'))
    plt.close()

    # MSE error plot
    plt.figure(figsize=(10, 6))
    plt.plot(mse_test, label='MSE Error')
    plt.title('MSE Error Over Test Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.savefig(os.path.join(model_directory, 'mse_error_plot.png'))
    plt.close()

    # Plot and save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, mse_test)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_directory, 'roc_curve.png'))
    plt.close()

    # Save the model
    autoencoder.save(os.path.join(model_directory, 'model.keras'))

    # Print evaluation metrics
    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"Accuracy: {accuracy_score(y_test, optimal_predictions)}")
    print(f"Precision: {precision_score(y_test, optimal_predictions)}")
    print(f"Recall: {recall_score(y_test, optimal_predictions)}")
    print(f"F1 Score: {f1_score(y_test, optimal_predictions)}")
    logging.info(f"Model Evaluation Metrics - Optimal Threshold: {optimal_threshold}, Accuracy: {accuracy_score(y_test, optimal_predictions)}, Precision: {precision_score(y_test, optimal_predictions)}, Recall: {recall_score(y_test, optimal_predictions)}, F1 Score: {f1_score(y_test, optimal_predictions)}")

def create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size):
    model_directory = os.path.join(root_path, f"model_ws{window_size}_ss{step_size}_ln{lstm_neurons}_e{epochs}_bs{batch_size}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    return model_directory
 
def prepare_data(normal_features, abnormal_features):
    # Combine normal and abnormal data
    X_combined = np.concatenate((normal_features, abnormal_features))
    y_combined = np.concatenate((np.zeros(len(normal_features)), np.ones(len(abnormal_features))))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Reshape data for LSTM input
    timesteps = 1  # Each audio is of length 1 mins
    n_features = X_train_scaled.shape[1]
    X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
    X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))

    return X_train_reshaped, X_val_reshaped, y_train, y_val

def hyperparameter_tuning(root_path, data_path, config_list, use_lstm=True):
    global SAMPLE_RATE  # global SAMPLE_RATE
    for config in config_list:
        window_size, step_size, lstm_neurons, epochs, batch_size = config
        logging.info(f"Config - Window: {window_size}s, Step: {step_size}s, LSTM Neurons: {lstm_neurons}, Epochs: {epochs}, Batch Size: {batch_size}")

        # Initialize lists to collect features and labels
        all_features = []
        all_labels = []

        # Process normal audio files
        audio_windows_generator = generate_audio_windows(data_path['normal'], window_size, step_size, sample_rate)
        for audio_windows in audio_windows_generator:
            normal_features = extract_features(audio_windows, sample_rate)
            all_features.extend(normal_features)
            all_labels.extend([0] * len(normal_features))  #  0 is the label for normal

        # Process abnormal audio files
        audio_windows_generator = generate_audio_windows(data_path['abnormal'], window_size, step_size, sample_rate)
        for audio_windows in audio_windows_generator:
            abnormal_features = extract_features(audio_windows, sample_rate)
            all_features.extend(abnormal_features)
            all_labels.extend([1] * len(abnormal_features))  #  1 is the label for abnormal

        # Convert lists to numpy arrays
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        # Combine, split, scale, and reshape data
        X_train, X_val, y_train, y_val = prepare_data(all_features, all_labels)

        # Create model directory
        model_directory = create_model_directory(root_path, window_size, step_size, lstm_neurons, epochs, batch_size)

        # Train model
        if use_lstm:
            autoencoder = simplified_autoencoder_with_lstm(X_train.shape[1], X_train.shape[2], lstm_neurons)
        else:
            autoencoder = simplified_autoencoder(X_train.shape[1], X_train.shape[2])

        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=1)

        # Evaluate model
        model_evaluation(autoencoder, X_val, y_val, model_directory)


def train_model(X_train, X_val):
    logging.info(f"Model training with data shape: {X_train.shape}")
    try:
        autoencoder = simplified_autoencoder_with_lstm(X_train.shape[1], X_train.shape[2])
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], verbose=1)
        logging.info("Model training completed")
    except Exception as e:
        logging.error("An error occurred during model training", exc_info=True)

def get_current_memory_usage():
    """
    Returns the current memory usage of the python process in megabytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to megabytes

def find_optimal_batch_size(data_path, window_size, step_size, test_batch_sizes):
    global SAMPLE_RATE
    memory_usage = {}
    performance = {}

    for batch_size in test_batch_sizes:
        try:
            # Start timer
            start_time = time.time()

            # Generate and process audio windows
            audio_windows_generator = generate_audio_windows(data_path['normal'], window_size, step_size, SAMPLE_RATE, batch_size=batch_size)
            for audio_windows in audio_windows_generator:
                _ = extract_features(audio_windows, SAMPLE_RATE) 

            #  memory usage and performance
            memory_usage[batch_size] = get_current_memory_usage()
            performance[batch_size] = time.time() - start_time

        except MemoryError:
            logging.warning(f"Batch size {batch_size} caused out-of-memory error.")
            break

    # Determine the optimal batch size based on highest performance without exceeding memory limit (#TODO)
    optimal_batch_size = max(performance, key=performance.get)
    return optimal_batch_size, memory_usage, performance

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
            (15, 7, 64, 100, 64),   
            (8, 4, 32, 40, 32),     
            (12, 6, 128, 50, 32),   
            (20, 10, 64, 75, 64),   
            (5, 2.5, 32, 100, 32),  
            (18, 9, 256, 60, 128), 
            (10, 5, 64, 150, 64),  
            (15, 5, 128, 100, 64),  
            (20, 10, 256, 80, 128), 
            (7, 3.5, 64, 50, 32),  
            (13, 6.5, 128, 120, 64)
        ]
        

        root_path = "Calf_Detection/Audio/Audio_Work_AE"
        # Hyperparameter tuning for LSTM Autoencoder
        hyperparameter_tuning(root_path, data_path, config_list, use_lstm=True)

        # Hyperparameter tuning for Dense Autoencoder
        hyperparameter_tuning(root_path, data_path, config_list, use_lstm=False)

        # test_batch_sizes = [10, 20, 30, 40, 50]  # Range of batch sizes to test
        # optimal_batch_size, memory_usage, performance = find_optimal_batch_size(data_path, 10, 5, test_batch_sizes)
        # print(f"Optimal Batch Size: {optimal_batch_size}")
        # print(f"Memory Usage (MB): {memory_usage}")
        # print(f"Performance (s): {performance}")

    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)