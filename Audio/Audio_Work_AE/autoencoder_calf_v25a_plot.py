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
import joblib
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LeakyReLU, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from logging import NullHandler
from tensorflow.keras.regularizers import L1L2
from sklearn.metrics import mean_squared_error



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
    {"window_size": 5, "step_size": 2.5, "expected_timesteps": 23, "lstm_neurons": 128, "epochs": 500, "batch_size": 30}
    # {"window_size": 10, "step_size": 5, "expected_timesteps": 11, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
    # {"window_size": 15, "step_size": 7.5, "expected_timesteps": 7, "lstm_neurons": 64, "epochs": 20, "batch_size": 32}
]

def normalize_features(features, scaler):
    return scaler.transform(features.reshape(-1, 1)).flatten()



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

def extract_features(audio, sample_rate, feature_extraction_logger,scaler_creater,output_dir):
    # Aggregate all feature extraction processes
    start_time = time.time()
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)    # Extract MFCCs
    mfccs_processed = np.mean(mfccs.T,axis=0)
    spectral_features = extract_spectral_features(audio, sample_rate, feature_extraction_logger)
    temporal_features = extract_temporal_features(audio, feature_extraction_logger)
    additional_features = extract_additional_features(audio, sample_rate, feature_extraction_logger)
    raw_audio_features = extract_raw_audio_features(audio, 10, feature_extraction_logger)
    features = np.concatenate((mfccs_processed,spectral_features, temporal_features, additional_features, raw_audio_features))
    if not scaler_creater:
        scaler_path = os.path.join(output_dir, "scaler.gz")
        scaler = joblib.load(scaler_path)
        # Reshape correctly for a single sample
        features = features.reshape(1, -1)  # Reshape for a single sample
        features = scaler.transform(features).flatten()  # Transform and then flatten back
    else:
        # No scaling when creating scaler
        pass
    if features.shape[0] != TOTAL_FEATURES:
        raise ValueError(f"Feature extraction error: Expected {TOTAL_FEATURES} features, got {features.shape[0]}")
    if feature_extraction_logger:
        feature_extraction_logger.info(f"Total feature extraction time: {convert_seconds(time.time() - start_time)}.")
        feature_extraction_logger.info(f"Feature shape: {features.shape}, Total feature extraction time: {convert_seconds(time.time() - start_time)}.")

    return features

def fit_scaler_to_training_data(training_paths, sample_rate, evaluation_directory):
    scaler = StandardScaler()
    features_list = []

    # Assume training_paths is a dictionary with paths to training data
    for label, path in training_paths.items():
        audio_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
        for file_path in audio_files[:100]:  # Limit to first 100 files or another representative set
            audio = load_audio_file(file_path, sample_rate)
            features = extract_features(audio, sample_rate,feature_extraction_logger=None,scaler_creater=True,output_dir=evaluation_directory)
            features_list.append(features)
    
    # Fit the scaler
    features_array = np.vstack(features_list)  # Convert list of arrays into a single 2D array
    scaler.fit(features_array)
    # Save the scaler for later use
    scaler_file_path = os.path.join(evaluation_directory, 'scaler.gz')
    try:
        joblib.dump(scaler, scaler_file_path)
        print(f"Scaler successfully saved to {scaler_file_path}")
    except Exception as e:
        print(f"Error saving the scaler: {e}")

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
                
# def build_autoencoder(expected_timesteps, total_features, lstm_neurons, evaluation_directory, load_weights):
#     model_file_path = os.path.join(evaluation_directory, "00models", "final_autoencoder_model.h5")
#     if load_weights and os.path.exists(model_file_path):
#         print(f"Loading model weights from {model_file_path}")
#         return load_model(model_file_path)
    
#     input_layer = Input(shape=(expected_timesteps, total_features))

#     # Encoder
#     # Using Conv1D with stride of 1 to maintain temporal resolution
#     x = Conv1D(64, kernel_size=3, padding='same', activation='relu', strides=1)(input_layer)
#     x = BatchNormalization()(x)
#     x = Dropout(0.1)(x)
#     x = LSTM(lstm_neurons, activation='tanh', return_sequences=True)(x)
#     x = LSTM(int(lstm_neurons / 2), activation='tanh', return_sequences=False)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.1)(x)
#     x = RepeatVector(expected_timesteps)(x)

#     # Decoder
#     x = LSTM(int(lstm_neurons / 2), activation='tanh', return_sequences=True)(x)
#     x = LSTM(lstm_neurons, activation='tanh', return_sequences=True)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.1)(x)
#     output_layer = TimeDistributed(Dense(total_features))(x)

#     autoencoder = Model(inputs=input_layer, outputs=output_layer)
#     autoencoder.compile(optimizer='adam', loss='mse')

#     return autoencoder
def build_autoencoder(expected_timesteps, total_features, lstm_neurons, evaluation_directory, load_weights):
    model_file_path = os.path.join(evaluation_directory, "00models", "final_autoencoder_model.h5")
    if load_weights and os.path.exists(model_file_path):
        print(f"Loading model weights from {model_file_path}")
        return load_model(model_file_path)
    
    input_layer = Input(shape=(expected_timesteps, total_features))

    # Encoder
    # Using Conv1D with stride of 1 to maintain temporal resolution
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu', strides=1)(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # Updated to Bidirectional LSTM
    x = Bidirectional(LSTM(lstm_neurons, activation='tanh', return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)
    x = Bidirectional(LSTM(int(lstm_neurons / 2), activation='tanh', return_sequences=False, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    x = RepeatVector(expected_timesteps)(x)

    # Decoder
    # Also updated to Bidirectional LSTMs
    x = Bidirectional(LSTM(int(lstm_neurons / 2), activation='tanh', return_sequences=True))(x)
    x = Bidirectional(LSTM(lstm_neurons, activation='tanh', return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    output_layer = TimeDistributed(Dense(total_features))(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss='mse')

    return autoencoder


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
                    features = extract_features(window, sample_rate, feature_extraction_logger=None,scaler_creater=False,output_dir=output_dir)
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

def write_rmse_to_file(batch_numbers, rmse_values, output_file):
    with open(output_file, 'w') as file:
        file.write('Batch Number, RMSE\n')  # Header
        for batch_number, rmse in zip(batch_numbers, rmse_values):
            file.write(f'{batch_number}, {rmse}\n')

def calculate_rmse(model, feature_dir):
    rmse_values = []
    batch_numbers = []  # Store batch numbers instead of full names
    for npz_file in sorted(os.listdir(feature_dir)):
        if not npz_file.endswith('.npz'):
            continue
        # Extract the batch number from the filename
        batch_number = int(npz_file.split('_')[1].split('.')[0])
        data_path = os.path.join(feature_dir, npz_file)
        data = np.load(data_path)
        features = data['features']  # shape: (1, expected_timesteps, total_features)
        
        # Predict
        predicted_features = model.predict(features,verbose=2)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(features.flatten(), predicted_features.flatten()))
        rmse_values.append(rmse)
        batch_numbers.append(batch_number)
    
    return batch_numbers, rmse_values

def plot_rmse(batch_names, rmse_values, evaluation_directory):
    plt.figure(figsize=(18, 6))
    plt.plot(batch_names, rmse_values, marker='o', linestyle='-', color='#04316A')
    plt.title('RMSE by Batch')
    plt.xlabel('Batch Number')
    plt.grid(True)
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_directory, f"rmse.png"))
    plt.close()

# =============== Experimentation Functions ===============

def experiment_with_configurations(evaluation_directory, hyperparameters_combinations,load_weights):
    for combination in hyperparameters_combinations:
        # Dataset directories
        train_dataset_dirname = f"ws{combination['window_size']}_ss{combination['step_size']}_et{combination['expected_timesteps']}_bs{combination['batch_size']}_test" # 09-10-2023
        val_dataset_dirname = f"ws{combination['window_size']}_ss{combination['step_size']}_et{combination['expected_timesteps']}_bs{combination['batch_size']}_val"
        
        train_feature_dir = os.path.join(evaluation_directory, train_dataset_dirname)
        val_feature_dir = os.path.join(evaluation_directory, val_dataset_dirname)

        # if not os.path.exists(train_feature_dir) or not os.path.exists(val_feature_dir):
        #     logging.error(f"One or both feature directories do not exist: {train_feature_dir}, {val_feature_dir}")
        #     continue
        
        # Load datasets
        train_dataset = create_dataset_from_npz(train_feature_dir, combination['expected_timesteps'], TOTAL_FEATURES, combination['batch_size'])
        val_dataset = create_dataset_from_npz(val_feature_dir, combination['expected_timesteps'], TOTAL_FEATURES, combination['batch_size'])
        
        # Build model
        model = build_autoencoder(combination['expected_timesteps'], TOTAL_FEATURES, combination['lstm_neurons'],evaluation_directory,load_weights)

        # Callbacks
        checkpoint_path = os.path.join(evaluation_directory,"00models/model_checkpoint.h5")
        early_stopping_callback = EarlyStopping(monitor='loss', patience=20, verbose=2, mode='min',min_delta=1e-4, restore_best_weights=True)
        callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='loss', mode='min'),
            ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.001, verbose=2),
            # early_stopping_callback
        ]
        
        # Training
        model.fit(train_dataset, epochs=combination['epochs'], callbacks=callbacks, verbose=2,validation_data=val_dataset)
        # model.fit(train_dataset, validation_data=val_dataset, epochs=combination['epochs'], callbacks=callbacks)
        
        # Save final model
        model_save_path = os.path.join(evaluation_directory,"00models/final_autoencoder_model.h5")
        model.save(model_save_path)
        print(f"Final model saved to {model_save_path}")
        

        model = load_model(model_save_path)
    
    # Directory containing your test features in batches
        test=f"ws{combination['window_size']}_ss{combination['step_size']}_et{combination['expected_timesteps']}_bs{combination['batch_size']}_test"
        val_feature_dir = os.path.join(evaluation_directory, test)
        test_feature_dir = os.path.join(evaluation_directory, val_feature_dir)
    
    # Calculate RMSE values for each test batch
        batch_numbers, rmse_values = calculate_rmse(model, test_feature_dir)
        
        sorted_pairs = sorted(zip(batch_numbers, rmse_values), key=lambda x: x[0])
        sorted_batch_numbers, sorted_rmse_values = zip(*sorted_pairs)
        
        output_file_path = os.path.join(evaluation_directory, "rmse_values.txt")
        write_rmse_to_file(sorted_batch_numbers, sorted_rmse_values, output_file_path)
    
    # Plot the RMSE values
        plot_rmse(sorted_batch_numbers, sorted_rmse_values, evaluation_directory)

def main(evaluation_directory, enable_logging):
    global LOGGING_ENABLED
    LOGGING_ENABLED = enable_logging
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    normal_paths = {'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_single_day'}
    validation_paths = {'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_single_day/09_Oct'}
    mode_1,mode_2,mode_3="train","val_true","test"
    
    ## Creating the standard scalar.
    # fit_scaler_to_training_data(normal_paths,SAMPLE_RATE,evaluation_directory)
    # Training creation
    # for combination in hyperparameters_combinations:
    #     save_features_in_batches(normal_paths, SAMPLE_RATE, combination, evaluation_directory, n_files_per_batch=30,mode=mode_2)
    #     print(f"Saved features in batches for combination: {combination}")   
        
    # # Validation creationCalf_Detection/Audio/Audio_Work_AE/autoencoder_calf_v25a.py
    # for combination in hyperparameters_combinations:
    #     save_features_in_batches(validation_paths, SAMPLE_RATE, combination, evaluation_directory, n_files_per_batch=30,mode=mode_3)
    #     print(f"Saved features in batches for combination: {combination}") 
        
    experiment_with_configurations(evaluation_directory, hyperparameters_combinations, load_weights=True)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v7'
    main(evaluation_directory, enable_logging=False)