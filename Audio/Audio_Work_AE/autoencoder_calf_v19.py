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
hyperparameters_combinations = [{"window_size_seconds": 7, "step_size_seconds": 3.5, "lstm_neurons": 64, "epochs": 20, "batch_size": 5}]

# Utility Functions
def create_model_directory(root_path, config):
    model_dir = os.path.join(root_path, "model_{}".format("_".join(map(str, config))))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

def load_and_concatenate_audio_files(file_paths, sample_rate):
    audio_streams = [load_audio_file(file_path, sample_rate) for file_path in file_paths]
    concatenated_stream = np.concatenate(audio_streams)
    return concatenated_stream

def calculate_expected_timesteps(window_size_seconds, step_size_seconds, total_audio_duration_seconds, sample_rate):
    window_size_samples = int(window_size_seconds * sample_rate)
    step_size_samples = int(step_size_seconds * sample_rate)
    total_duration_samples = int(total_audio_duration_seconds * sample_rate)
    return (total_duration_samples - window_size_samples) // step_size_samples + 1

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

def modified_audio_data_generator(paths, sample_rate, window_size_seconds, step_size_seconds, total_features, batch_size, expected_timesteps):
    window_size_samples = int(window_size_seconds * sample_rate)
    step_size_samples = int(step_size_seconds * sample_rate)
    
    def generator():
        file_paths = [os.path.join(paths[label], filename) for label in paths for filename in os.listdir(paths[label]) if filename.endswith('.wav')]
        batch_count = 0
        while batch_count * batch_size < len(file_paths):  # Ensure processing continues until all files are used
            start_index = batch_count * batch_size
            end_index = start_index + batch_size
            batch_file_paths = file_paths[start_index:end_index]
            batch_audio_data = []
            for fp in batch_file_paths:
                audio = load_audio_file(fp, sample_rate)
                batch_audio_data.append(audio)
            concatenated_audio = np.concatenate(batch_audio_data, axis=0)
            
            # Apply sliding window
            windows = []
            for start in range(0, len(concatenated_audio) - window_size_samples + 1, step_size_samples):
                end = start + window_size_samples
                window = concatenated_audio[start:end]
                windows.append(window)
            
            # Extract features for each window
            features = np.array([extract_features(window, sample_rate) for window in windows])
            
            # Check and adjust the shape of the features to match expected_timesteps if necessary
            if features.shape[0] < expected_timesteps:
                # Pad the features if there are fewer windows than expected
                padding = np.zeros((expected_timesteps - features.shape[0], total_features))
                features = np.vstack((features, padding))
            elif features.shape[0] > expected_timesteps:
                # Truncate features if there are more windows than expected
                features = features[:expected_timesteps, :]
            
            features = features.reshape(-1, expected_timesteps, total_features)  # Reshape for the expected model input
            yield features, features  # Yielding features as both inputs and targets for an autoencoder
            
            batch_count += 1

    return generator

def create_modified_tf_dataset(paths, sample_rate, window_size_seconds, step_size_seconds, total_features, batch_size, expected_timesteps):
    # Create a dataset using the generator
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: modified_audio_data_generator(paths, sample_rate, window_size_seconds, step_size_seconds, total_features, batch_size, expected_timesteps),
        output_signature=(
            tf.TensorSpec(shape=(None, expected_timesteps, total_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, expected_timesteps, total_features), dtype=tf.float32)
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

def build_autoencoder(expected_timesteps,total_features, lstm_neurons):
    input_shape = (expected_timesteps, total_features)
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(filters=32, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = LSTM(lstm_neurons, return_sequences=True)(x)
    x = Bidirectional(LSTM(int(lstm_neurons/2), return_sequences=False))(x)

    # Bottleneck
    x = RepeatVector(input_shape[0])(x)

    # Decoder
    x = Bidirectional(LSTM(int(lstm_neurons/2), return_sequences=True))(x)
    x = LSTM(lstm_neurons, return_sequences=True)(x)
    x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    output_layer = TimeDistributed(Dense(input_shape[1]))(x)

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
    for combination in hyperparameters_combinations:
        window_size_seconds = combination["window_size_seconds"]
        step_size_seconds = combination["step_size_seconds"]
        batch_size = combination["batch_size"]
        lstm_neurons = combination["lstm_neurons"]
        epochs = combination["epochs"]

        # Calculate total_audio_duration_seconds based on batch_size and the length of each audio file
        total_audio_duration_seconds = 60 * batch_size  # Assuming each audio file is 1 minute long
        expected_timesteps = calculate_expected_timesteps(window_size_seconds, step_size_seconds, total_audio_duration_seconds, sample_rate)
        
        # Log the calculated expected timesteps for debugging
        logging.info(f"Calculated expected timesteps: {expected_timesteps}")

        # Adjust the creation of the training dataset to use the calculated expected_timesteps
        train_dataset = create_modified_tf_dataset(paths, sample_rate, window_size_seconds, step_size_seconds, TOTAL_FEATURES, batch_size, expected_timesteps)

        # Now build the autoencoder model using the calculated expected_timesteps
        autoencoder = build_autoencoder(expected_timesteps, TOTAL_FEATURES, lstm_neurons)

        # Train the autoencoder
        logging.info("Starting training...")
        autoencoder.fit(train_dataset, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=3)])
        
        # Save the trained model
        model_save_dir = os.path.join(evaluation_directory, f"model_ws{window_size_seconds}_ss{step_size_seconds}_ln{lstm_neurons}_ep{epochs}_bs{batch_size}")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_path = os.path.join(model_save_dir, "autoencoder_model.h5")
        autoencoder.save(model_path)
        logging.info(f"Model saved to {model_path}")

        # Create test dataset using abnormal data
        # test_dataset = create_modified_tf_dataset(
        #     {'abnormal': abnormal_path}, sample_rate, window_size, step_size, TOTAL_FEATURES, batch_size # Batch_size
        # )

        # Evaluate the model on the abnormal test dataset
        # model_evaluation(autoencoder, test_dataset, model_save_dir, combination)

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