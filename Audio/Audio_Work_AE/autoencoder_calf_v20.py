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
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def concatenated_features_generator(test_files, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch=30):
    """Yields concatenated features from n_files_per_batch audio files."""
    for i in range(0, len(test_files), n_files_per_batch):
        batch_files = test_files[i:i + n_files_per_batch]
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

def create_evaluation_dataset(generator, output_shapes, output_types=tf.float32, batch_size=None):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=output_types,
        output_shapes=output_shapes
    )
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)

def evaluate_model(model, dataset):
    all_batch_mse = []
    for batch_features in dataset:
        reconstructed = model.predict(batch_features)
        mse = np.mean(np.square(batch_features - reconstructed), axis=(1, 2))
        batch_mse = np.mean(mse)
        all_batch_mse.append(batch_mse)
        print(f"Batch MSE: {batch_mse}")
    
    overall_mse = np.mean(all_batch_mse)
    print(f"Overall MSE: {overall_mse}")
    return all_batch_mse, overall_mse
   

def test_model(test_files, model, sample_rate, window_size, step_size, expected_timesteps, total_features):
    test_features = []
    test_labels = []

    for file_path in test_files:
        audio = load_audio_file(file_path, sample_rate)
        windows = sliding_window(audio, window_size, step_size, sample_rate)
        
        for start in range(0, len(windows) - expected_timesteps + 1, expected_timesteps):
            segment_windows = windows[start:start + expected_timesteps]
            if len(segment_windows) == expected_timesteps:
                features = np.array([extract_features(window, sample_rate) for window in segment_windows])
                test_features.append(features)
                test_labels.append(file_path)  # 

    test_features = np.array(test_features)  # Shape: (num_sequences, expected_timesteps, total_features)
    
    # Model prediction
    reconstructed_features = model.predict(test_features)

    # Evaluate performance (e.g., MSE)
    mse = np.mean(np.square(test_features - reconstructed_features), axis=-1)
    average_mse = np.mean(mse)

    print(f"Average MSE: {average_mse}")
    return mse, test_labels

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

        # Create training dataset
        # train_dataset = create_tf_dataset(
        #     paths, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        # )

        # Build and train the model
        # autoencoder = build_autoencoder(expected_timesteps, TOTAL_FEATURES, lstm_neurons)
        # autoencoder.fit(train_dataset, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=3)])
        
        #Save the model
        # model_path = os.path.join(model_save_dir, "autoencoder_model.h5")
        # autoencoder.save(model_path)
        # logging.info(f"Model saved to {model_path}")

        model_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3/ws7_ss3.5_et10_lstm128_1epochs_bs32/autoencoder_model.h5"
        autoencoder=load_model(model_path)
        # Create test dataset using abnormal data
        # test_dataset = create_tf_dataset(
        #     {'abnormal': abnormal_path}, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        # )

        test_files=['/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_superset/output_2023-10-08_16-23-35.wav']
        # mse, test_labels = test_model(test_files, autoencoder, SAMPLE_RATE, window_size, step_size, expected_timesteps, TOTAL_FEATURES)
        # Create the generator for batches of files
        generator = lambda: concatenated_features_generator(
            abnormal_path, SAMPLE_RATE, window_size, step_size, expected_timesteps, TOTAL_FEATURES, n_files_per_batch=2
        )

        # Create TensorFlow Dataset
        dataset = create_evaluation_dataset(
            generator,
            output_shapes=(None, expected_timesteps, TOTAL_FEATURES)
        )

        # Evaluate the model
        batch_mse, overall_mse = evaluate_model(autoencoder, dataset)


        
        
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