import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import librosa
import os
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, LeakyReLU, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model, load_model


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class ResetStatesCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()


# Constants
SAMPLE_RATE = 16000
TOTAL_FEATURES = 23
# hyperparameters_combinations = [
#     {"window_size": 7, "step_size": 3.5, "expected_timesteps": 10, "lstm_neurons": 128, "epochs": 20, "batch_size": 32},
#     {"window_size": 15, "step_size": 7.5, "expected_timesteps": 10, "lstm_neurons": 256, "epochs": 20, "batch_size": 32}
# ]
hyperparameters_combinations = [
    {"window_size": 7, "step_size": 3.5, "expected_timesteps": 10, "lstm_neurons": 128, "epochs": 1, "batch_size": 32}
]

# Utility Functions
def create_model_directory(root_path, config):
    model_dir = os.path.join(root_path, "model_{}".format("_".join(map(str, config))))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

# Feature Extraction:
def extract_spectral_features(audio, sample_rate):# Spectral Features 
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

def extract_temporal_features(audio):# Temporal Features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

def extract_additional_features(audio, sample_rate):#  Additional features
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)

def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)    # Extract MFCCs
    mfccs_processed = np.mean(mfccs.T,axis=0)
    
    spectral_features = extract_spectral_features(audio, sample_rate)    # Extract additional features
    temporal_features = extract_temporal_features(audio)
    additional_features = extract_additional_features(audio, sample_rate)
    
    features = np.concatenate((mfccs_processed, spectral_features, temporal_features, additional_features))    # Combine all features
    return features

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []

    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
    return windows

def audio_data_generator(paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch=30):
    def generator():
        for label, path in paths.items():
            file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
            batch_audio = []
            
            for i, file_path in enumerate(file_paths):
                audio = load_audio_file(file_path, sample_rate)
                batch_audio.append(audio)
                
                # Process in batches of n_files_per_batch or when it's the last file
                if (i + 1) % n_files_per_batch == 0 or i == len(file_paths) - 1:
                    concatenated_audio = np.concatenate(batch_audio)
                    batch_audio = []  # Reset for next batch
                    
                    windows = sliding_window(concatenated_audio, window_size, step_size, sample_rate)
                    for start in range(0, len(windows) - expected_timesteps + 1, expected_timesteps):
                        segment_windows = windows[start:start + expected_timesteps]
                        if len(segment_windows) == expected_timesteps:
                            features = [extract_features(window, sample_rate) for window in segment_windows]
                            features = np.stack(features)  # Shape: (expected_timesteps, total_features)
                            yield features, features

    return generator

def create_tf_dataset(paths, sample_rate, window_size, step_size, expected_timesteps, total_features, batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        generator=audio_data_generator(paths, sample_rate, window_size, step_size, expected_timesteps, total_features),
        output_signature=(
            tf.TensorSpec(shape=(expected_timesteps, total_features), dtype=tf.float32),
            tf.TensorSpec(shape=(expected_timesteps, total_features), dtype=tf.float32),
        )
    )
    return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

def dynamic_test_files_generator(directory_path, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch=30):
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.wav')]
    files.sort()  # Should check for training
    
    for i in range(0, len(files), n_files_per_batch):
        batch_files = files[i:i + n_files_per_batch]
        concatenated_audio = np.array([])

        for file_path in batch_files:
            audio = load_audio_file(file_path, sample_rate)
            concatenated_audio = np.concatenate((concatenated_audio, audio))

        windows = sliding_window(concatenated_audio, window_size, step_size, sample_rate)
        batch_features = []
        for start in range(0, len(windows) - expected_timesteps + 1, expected_timesteps):
            segment_windows = windows[start:start + expected_timesteps]
            if len(segment_windows) == expected_timesteps:
                features = np.array([extract_features(window, sample_rate) for window in segment_windows])
                batch_features.append(features)
                
        if batch_features:
            yield np.array(batch_features)

def create_evaluation_dataset_from_directory(directory_path, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch=30):
    generator = lambda: dynamic_test_files_generator(
        directory_path, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch
    )
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(None, expected_timesteps, total_features), dtype=tf.float32)
    ).prefetch(tf.data.AUTOTUNE)

def evaluate_model(model, dataset):
    all_batch_mse = []
    batch_number = 0  # Initialize batch counter
    
    for batch_features in dataset:
        reconstructed = model.predict(batch_features)
        mse = np.mean(np.square(batch_features - reconstructed), axis=(1, 2))
        batch_mse = np.mean(mse)
        all_batch_mse.append(batch_mse)
        
        print(f"Batch #{batch_number} MSE: {batch_mse}")
        
        batch_number += 1  
    
    overall_mse = np.mean(all_batch_mse)
    print(f"Overall MSE: {overall_mse}")
    return all_batch_mse, overall_mse

def build_autoencoder(expected_timesteps, total_features, lstm_neurons, batch_size):  
    # Input layer adjusted for stateful LSTMs
    input_layer = Input(batch_shape=(batch_size, expected_timesteps, total_features))

    # Encoder with stateful LSTM
    x = Conv1D(filters=32, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = LSTM(lstm_neurons, return_sequences=False, stateful=True)(x)  # Note the change here
    
    # Replicate encoder output for decoder input
    x = RepeatVector(expected_timesteps)(x)
    
    # Decoder
    x = LSTM(lstm_neurons, return_sequences=True, stateful=True)(x)
    x = TimeDistributed(Dense(total_features))(x)
    
    autoencoder = Model(inputs=input_layer, outputs=x)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def hyperparameter_tuning(root_path, paths, sample_rate, evaluation_directory, hyperparameters_combinations):
    abnormal_path = paths['abnormal']  # Path to the abnormal data
    print(f"Type is :{type(abnormal_path)}")
    print(abnormal_path)

    for combination in hyperparameters_combinations:
        # Hyperparameter extraction
        window_size = combination["window_size"]
        step_size = combination["step_size"]
        expected_timesteps = combination["expected_timesteps"]
        lstm_neurons = combination["lstm_neurons"]
        epochs = combination["epochs"]
        batch_size = combination["batch_size"]  

        dataset_dirname = f"ws{window_size}_ss{step_size}_et{expected_timesteps}_lstm{lstm_neurons}_{epochs}epochs_bs{batch_size}"
        model_save_dir = os.path.join(evaluation_directory, dataset_dirname)
        os.makedirs(model_save_dir, exist_ok=True)

        #   Create training dataset
        train_dataset = create_tf_dataset(
            paths, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        )

        #   Build and train the model
        autoencoder = build_autoencoder(expected_timesteps, TOTAL_FEATURES, lstm_neurons,   batch_size)
        autoencoder.fit(train_dataset, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=3),ResetStatesCallback()])
        
        #   Save the model
        model_path = os.path.join(model_save_dir, "autoencoder_model.h5")
        autoencoder.save(model_path)
        logging.info(f"Model saved to {model_path}")

        model_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3/ws7_ss3.5_et10_lstm128_1epochs_bs32/autoencoder_model.h5"
        autoencoder=load_model(model_path)
        # Create test dataset using abnormal data
        # test_dataset = create_tf_dataset(
        #     {'abnormal': abnormal_path}, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        # )

        # Create the generator for batches of files
        test_dataset = create_evaluation_dataset_from_directory(
            abnormal_path, SAMPLE_RATE, window_size, step_size, expected_timesteps, TOTAL_FEATURES, n_files_per_batch=2
        )
        # Evaluate the model
        # batch_mse, overall_mse = evaluate_model(autoencoder, test_dataset)

def main(evaluation_directory):
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    paths = {
        'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset',
        'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset'
    }
    hyperparameter_tuning(root_path, paths, SAMPLE_RATE, evaluation_directory, hyperparameters_combinations)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3'
    main(evaluation_directory)