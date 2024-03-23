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
from sklearn.preprocessing import StandardScaler
from logging import NullHandler
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LeakyReLU, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class ResetStatesCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()

# =============== Setup and Configuration ===============

SAMPLE_RATE = 44100
TOTAL_FEATURES = 33
LOGGING_ENABLED = False
log_dir = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/logs"

# Setup a global logging directory
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

# =============== Helper Functions ===============

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

hyperparameters_combinations = [
    {"window_size": 5, "step_size": 2.5, "expected_timesteps": 23, "lstm_neurons": 128, "epochs": 100, "batch_size": 30}
    # {"window_size": 10, "step_size": 5, "expected_timesteps": 11, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
    # {"window_size": 15, "step_size": 7.5, "expected_timesteps": 7, "lstm_neurons": 64, "epochs": 20, "batch_size": 32}
]

# =============== Audio Processing Functions ===============

def load_audio_file(file_path, sample_rate=SAMPLE_RATE, logger=None):
    start_time = time.time()
    audio, _ = librosa.load(file_path, sr=sample_rate)
    if logger:
        logger.info(f"Loaded audio file {file_path} in {convert_seconds(time.time() - start_time)}.")
    return audio

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []

    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
    return windows

# =============== Feature Extraction Functions ===============

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
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    if features.shape[0] != TOTAL_FEATURES:
        raise ValueError(f"Feature extraction error: Expected {TOTAL_FEATURES} features, got {features.shape[0]}")
    if feature_extraction_logger:
        feature_extraction_logger.info(f"Total feature extraction time: {convert_seconds(time.time() - start_time)}.")
        feature_extraction_logger.info(f"Feature shape: {features.shape}, Total feature extraction time: {convert_seconds(time.time() - start_time)}.")

    return features

def adjust_features_shape(features, expected_timesteps, total_features):

    if len(features) < expected_timesteps:
        # Pad with zeros
        padding = np.zeros((expected_timesteps - len(features), total_features))
        features = np.vstack((features, padding))
    elif len(features) > expected_timesteps:
        # Truncate to expected_timesteps
        features = features[:expected_timesteps]
    return features

# =============== Model Building Functions ===============
                
def build_autoencoder(expected_timesteps, total_features,lstm_neurons):
    input_layer = Input(shape=(expected_timesteps, total_features))

    # Encoder
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = LSTM(64, activation='tanh', return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = RepeatVector(expected_timesteps)(x)

    # Decoder
    x = LSTM(64, activation='tanh', return_sequences=True)(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    output_layer = TimeDistributed(Dense(total_features))(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder
# def build_autoencoder(expected_timesteps, total_features, lstm_neurons):
#     input_layer = Input(shape=(expected_timesteps, total_features))
    
#     # Encoder with Conv1D and increased complexity
#     x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
#     x = MaxPooling1D(2, padding='same')(x)
#     x = LSTM(lstm_neurons, activation='tanh', return_sequences=True, recurrent_dropout=0.2)(x)
#     x = LSTM(lstm_neurons // 2, activation='tanh', return_sequences=False, recurrent_dropout=0.2)(x)
#     x = RepeatVector(expected_timesteps)(x)

#     # Decoder
#     x = LSTM(lstm_neurons // 2, activation='tanh', return_sequences=True, recurrent_dropout=0.2)(x)
#     x = LSTM(lstm_neurons, activation='tanh', return_sequences=True, recurrent_dropout=0.2)(x)
#     x = TimeDistributed(Dense(total_features))(x)

#     autoencoder = Model(inputs=input_layer, outputs=output_layer)
#     autoencoder.compile(optimizer='adam', loss='mse')
#     return autoencoder


# =============== Data Preparation Functions ===============

def save_features_in_batches(paths, sample_rate, combination, output_dir, n_files_per_batch, mode):
    window_size = combination["window_size"]
    step_size = combination["step_size"]
    expected_timesteps = combination["expected_timesteps"]
    batch_size = combination["batch_size"]
    
    feature_save_dir = os.path.join(output_dir, f"ws{window_size}_ss{step_size}_et{expected_timesteps}_bs{batch_size}_{mode}")
    os.makedirs(feature_save_dir, exist_ok=True)

    batch_counter = 0
    sequence_features = []  # Accumulate features here

    for label, path in paths.items():
        file_paths = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.wav')]
        
        # Process in specified batch sizes
        for start in range(0, len(file_paths), n_files_per_batch):
            batch_file_paths = file_paths[start:start + n_files_per_batch]
            
            for file_path in batch_file_paths:
                audio = load_audio_file(file_path, sample_rate)
                windows = sliding_window(audio, window_size, step_size, sample_rate)

                for window in windows:
                    features = extract_features(window, sample_rate)
                    sequence_features.append(features)
                    
                    # Check if we have enough features for a complete sequence
                    if len(sequence_features) >= expected_timesteps:
                        batch_features = np.array(sequence_features[:expected_timesteps])
                        sequence_features = sequence_features[expected_timesteps:]  # Remove used features
                        
                        # Save the batch if it matches the expected size
                        if batch_features.shape == (expected_timesteps, TOTAL_FEATURES):
                            np.savez_compressed(os.path.join(feature_save_dir, f"batch_{batch_counter}.npz"), features=batch_features.reshape(1, expected_timesteps, TOTAL_FEATURES))
                            batch_counter += 1
                        else:
                            print("Skipped batch due to incorrect shape.")
            
            # Handle leftovers for each n_files_per_batch cycle
            if sequence_features:
                # Pad the leftover features to match the expected timesteps
                while len(sequence_features) < expected_timesteps:
                    sequence_features.append(np.zeros(TOTAL_FEATURES))  # Padding with zeros
                    
                leftover_features = np.array(sequence_features).reshape(1, expected_timesteps, TOTAL_FEATURES)
                np.savez_compressed(os.path.join(feature_save_dir, f"batch_{batch_counter}.npz"), features=leftover_features)
                batch_counter += 1
                sequence_features = []  # Reset for the next batch of files

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
                       # handle features with incorrect shapes
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

# =============== Testing Functions ===============

def load_features_as_dataset(feature_dir, expected_timesteps, total_features, batch_size=1):

    def generator():
        for npz_file in sorted(os.listdir(feature_dir)):
            if npz_file.endswith('.npz'):
                data = np.load(os.path.join(feature_dir, npz_file))
                features = data['features']
                yield features
                
    output_signature = tf.TensorSpec(shape=(None, expected_timesteps, total_features), dtype=tf.float32)
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(batch_size)
    return dataset

def test_consecutive_anomaly_per_window(model, feature_dir, threshold, min_consecutive_anomalies=2):
    anomaly_results = []
    for npz_file in sorted(os.listdir(feature_dir)):
        if not npz_file.endswith('.npz'):
            continue
        data_path = os.path.join(feature_dir, npz_file)
        data = np.load(data_path)
        features = data['features']  # shape : (1, expected_timesteps, total_features)
        
        # Predict and calculate MSE
        reconstructed = model.predict(features)
        mse = np.mean(np.square(features - reconstructed), axis=-1)
        mse = np.squeeze(mse)  # Removing single dimensions

        # Detect consecutive anomalies
        consecutive_count = 0
        start_index = None
        for i, error in enumerate(mse):
            if error > threshold:
                consecutive_count += 1
                if start_index is None:
                    start_index = i
                if consecutive_count >= min_consecutive_anomalies:
                    anomaly_results.append((npz_file, start_index, i, True))
                    consecutive_count = 0  # Reset count
                    start_index = None  # Reset start index
            else:
                consecutive_count = 0  # Reset count
                start_index = None  # Reset start index
    
    return anomaly_results

# =============== Experimentation Functions ===============

def experiment_with_configurations(evaluation_directory, hyperparameters_combinations):
    for combination in hyperparameters_combinations:
        # Dataset directories
        train_dataset_dirname = f"ws{combination['window_size']}_ss{combination['step_size']}_et{combination['expected_timesteps']}_bs{combination['batch_size']}_train"
        val_dataset_dirname = f"ws{combination['window_size']}_ss{combination['step_size']}_et{combination['expected_timesteps']}_bs{combination['batch_size']}_val"
        
        train_feature_dir = os.path.join(evaluation_directory, train_dataset_dirname)
        val_feature_dir = os.path.join(evaluation_directory, val_dataset_dirname)

        if not os.path.exists(train_feature_dir) or not os.path.exists(val_feature_dir):
            logging.error(f"One or both feature directories do not exist: {train_feature_dir}, {val_feature_dir}")
            continue
        
        # Load datasets
        train_dataset = create_dataset_from_npz(train_feature_dir, combination['expected_timesteps'], TOTAL_FEATURES, combination['batch_size'])
        val_dataset = create_dataset_from_npz(val_feature_dir, combination['expected_timesteps'], TOTAL_FEATURES, combination['batch_size'])
        
        # Build model
        model = build_autoencoder(combination['expected_timesteps'], TOTAL_FEATURES, combination['lstm_neurons'])

        # Callbacks
        checkpoint_path = os.path.join(evaluation_directory,"00models/model_checkpoint.h5")
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
        ]
        
        # Training
        model.fit(train_dataset, validation_data=val_dataset, epochs=combination['epochs'], callbacks=callbacks)
        
        # Save final model
        model_save_path = os.path.join(evaluation_directory,"00models/final_autoencoder_model.h5")
        model.save(model_save_path)
        print(f"Final model saved to {model_save_path}")
        
        # Validation set.
        model = tf.keras.models.load_model(model_save_path)
        feature_dir = val_feature_dir
        thresholds = [0.60, 0.70, 0.80]        
        min_consecutive_anomalies_list = [2, 3, 4, 5]

        # for threshold in thresholds:
        #     for min_consecutive_anomalies in min_consecutive_anomalies_list:
        #         print(f"Testing with threshold: {threshold}, Min consecutive anomalies: {min_consecutive_anomalies}")
        #         anomaly_results = test_consecutive_anomaly_per_window(model, feature_dir, threshold, min_consecutive_anomalies)
        #         for result in anomaly_results:
        #             print(f"File: {result[0]}, Start Window: {result[1]}, End Window: {result[2]}, Anomaly Detected: {result[3]}")

def main(evaluation_directory, enable_logging):
    global LOGGING_ENABLED
    LOGGING_ENABLED = enable_logging
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    normal_paths = {'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_training_set'}
    validation_paths = {'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_validation_set'}
    mode_1,mode_2,mode_3="train","val","test"
    
    # Training creation
    # for combination in hyperparameters_combinations:
    #     save_features_in_batches(normal_paths, SAMPLE_RATE, combination, evaluation_directory, n_files_per_batch=30,mode=mode_1)
    #     print(f"Saved features in batches for combination: {combination}")   
        
    # Validation creation
    # for combination in hyperparameters_combinations:
    #     save_features_in_batches(validation_paths, SAMPLE_RATE, combination, evaluation_directory, n_files_per_batch=30,mode=mode_2)
    #     print(f"Saved features in batches for combination: {combination}") 
        
    experiment_with_configurations(evaluation_directory, hyperparameters_combinations)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug'
    main(evaluation_directory, enable_logging=False)