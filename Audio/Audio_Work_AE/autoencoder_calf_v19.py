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
from tensorflow.keras.models import Model


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE = 44100
TOTAL_FEATURES = 23
hyperparameters_combinations = [
    {"window_size": 0.5, "step_size": 0.25, "expected_timesteps": 10, "lstm_neurons": 128, "epochs":2,"batch_size":32},
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

def audio_data_generator(paths, sample_rate, window_size, step_size, expected_timesteps, total_features):
    def generator():
        for label, path in paths.items():
            for filename in os.listdir(path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(path, filename)
                    audio = load_audio_file(file_path, sample_rate)
                    windows = sliding_window(audio, window_size, step_size, sample_rate)
                    if len(windows) >= expected_timesteps:
                        features = [extract_features(window, sample_rate) for window in windows[:expected_timesteps]]
                        features = np.stack(features)  # (EXPECTED_TIMESTEPS, TOTAL_FEATURES)
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
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_autoencoder(expected_timesteps, total_features, lstm_neurons):  
    input_shape = (expected_timesteps, total_features)
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(filters=32, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = LSTM(lstm_neurons, return_sequences=True)(x)
    x = Bidirectional(LSTM(int(lstm_neurons/2), return_sequences=False))(x)
    
    # Bottleneck
    x = RepeatVector(expected_timesteps)(x)
    
    # Decoder
    x = Bidirectional(LSTM(int(lstm_neurons/2), return_sequences=True))(x)
    x = LSTM(lstm_neurons, return_sequences=True)(x)
    
    x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    output_layer = TimeDistributed(Dense(total_features))(x)
    
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def model_evaluation(model, test_dataset, evaluation_directory, combination):
    mse_test = []
    for features, labels in test_dataset:
        predictions = model.predict(features)
        mse = np.mean(np.power(features.numpy() - predictions, 2), axis=(1, 2))
        mse_test.extend(mse)
    
    # Calculate the average MSE for the test dataset
    average_mse = np.mean(mse_test)
    logging.info(f"Average MSE on test dataset: {average_mse}")

    # Save MSE results
    model_config = f"ws{combination['window_size']}_ss{combination['step_size']}_et{combination['expected_timesteps']}_lstm{combination['lstm_neurons']}_{combination['epochs']}epochs"
    mse_filename = os.path.join(evaluation_directory, f"{model_config}_mse_test_results.txt")
    with open(mse_filename, "w") as file:
        file.write(f"Average MSE: {average_mse}\n")

    logging.info(f"MSE results saved to {mse_filename}")

def hyperparameter_tuning(root_path, paths, sample_rate, evaluation_directory, hyperparameters_combinations):
    abnormal_path = paths['abnormal']  # Path to the abnormal data

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

        # Create training dataset
        train_dataset = create_tf_dataset(
            paths, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        )

        # Build and train the model
        autoencoder = build_autoencoder(expected_timesteps, TOTAL_FEATURES, lstm_neurons)
        autoencoder.fit(train_dataset, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=3)])
        
        # Save the model
        model_path = os.path.join(model_save_dir, "autoencoder_model.h5")
        autoencoder.save(model_path)
        logging.info(f"Model saved to {model_path}")

        # Create test dataset using abnormal data
        test_dataset = create_tf_dataset(
            {'abnormal': abnormal_path}, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        )

        # Evaluate the model on the abnormal test dataset
        model_evaluation(autoencoder, test_dataset, model_save_dir, combination)

def main(evaluation_directory):
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    paths = {
        'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_superset',
        'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_superset'
    }
    hyperparameter_tuning(root_path, paths, SAMPLE_RATE, evaluation_directory, hyperparameters_combinations)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3'
    main(evaluation_directory)