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

def rebuild_model_for_prediction(expected_timesteps, total_features, lstm_neurons):
    # Note: No batch size specified here, using None to allow flexibility.
    input_layer = Input(batch_shape=(1, expected_timesteps, total_features))

    # Reconstruct your model architecture here
    x = Conv1D(filters=32, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = LSTM(lstm_neurons, return_sequences=False, stateful=True)(x)
    
    x = RepeatVector(expected_timesteps)(x)
    
    x = LSTM(lstm_neurons, return_sequences=True, stateful=True)(x)
    x = TimeDistributed(Dense(total_features))(x)
    
    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    
    return model

def test_model_on_audio_file(model_path, audio_file_path, sample_rate, window_size, step_size, expected_timesteps, total_features, txt_file_path):
    # Rebuild model for prediction with batch_size=1
    model = rebuild_model_for_prediction(expected_timesteps, total_features, lstm_neurons=128)
    model.load_weights(model_path)

    # Process the audio file and predict
    audio = load_audio_file(audio_file_path, sample_rate)
    windows = sliding_window(audio, window_size, step_size, sample_rate)
    windows_features_list = [extract_features(window, sample_rate) for window in windows]
    
    # Ensure there's enough windows to form at least one sequence of expected_timesteps
    if len(windows_features_list) < expected_timesteps:
        print("Not enough data for a full sequence.")
        return

    # Organize windows into sequences
    sequences = [windows_features_list[i:i + expected_timesteps] for i in range(0, len(windows_features_list), expected_timesteps) if len(windows_features_list[i:i + expected_timesteps]) == expected_timesteps]
    sequences_features = np.array(sequences)

    # Initialize an empty list to store reconstruction errors for each sequence
    sequence_reconstruction_errors = []

    # Predict and calculate reconstruction error for each sequence
    for sequence in sequences_features:
        sequence_reshaped = sequence.reshape(1, expected_timesteps, total_features)
        predicted_sequence = model.predict(sequence_reshaped, batch_size=1)
        sequence_error = np.mean(np.power(sequence_reshaped - predicted_sequence, 2), axis=(1, 2))
        sequence_reconstruction_errors.extend(sequence_error)

    # Save the reconstruction errors for each sequence
    np.savetxt(txt_file_path, sequence_reconstruction_errors, fmt='%f')
    print(f"Reconstruction errors for each sequence saved to {txt_file_path}")
    
def test_audio_data_generator(file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch=30):
    # Generator to load and concatenate audio files, then yield sequences
    batch_audio = []
    for i, file_path in enumerate(file_paths):
        audio = load_audio_file(file_path, sample_rate)
        batch_audio.append(audio)
        # Once enough files are loaded or it's the last file, process them
        if (i + 1) % n_files_per_batch == 0 or i == len(file_paths) - 1:
            concatenated_audio = np.concatenate(batch_audio)
            batch_audio = []  # Reset for the next batch
            
            windows = sliding_window(concatenated_audio, window_size, step_size, sample_rate)
            sequences = [windows[j:j + expected_timesteps] for j in range(len(windows) - expected_timesteps + 1)]
            for sequence in sequences:
                features_sequence = np.array([extract_features(window, sample_rate) for window in sequence])
                yield features_sequence.reshape(1, expected_timesteps, total_features)

def predict_and_store_errors(model, file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch, output_file):
    generator = test_audio_data_generator(file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch)
    with open(output_file, 'w') as f:
        for sequence in generator:
            predicted_sequence = model.predict(sequence)
            error = np.mean(np.power(sequence - predicted_sequence, 2))
            f.write(f"{error}\n")


def hyperparameter_tuning(root_path, paths, sample_rate, evaluation_directory, hyperparameters_combinations):
    abnormal_path = paths['abnormal']  # Path to the abnormal data
    normal_path=paths['normal']
    # List of file paths
    file_paths = [os.path.join(abnormal_path, f) for f in os.listdir(abnormal_path) if f.endswith('.wav')]
    
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
            {'normal': normal_path}, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        )

        #   Build and train the model
        autoencoder = build_autoencoder(expected_timesteps, TOTAL_FEATURES, lstm_neurons,   batch_size)
        autoencoder.fit(train_dataset, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=3),ResetStatesCallback()])
        
        #   Save the model
        model_path = os.path.join(model_save_dir, "autoencoder_model.h5")
        autoencoder.save(model_path)
        logging.info(f"Model saved to {model_path}")
        
        #   Rebuild for testing.        
        new_model=rebuild_model_for_prediction(expected_timesteps,TOTAL_FEATURES,lstm_neurons)
        new_model.load_weights(model_path)
        txt_file_path=os.path.join(model_save_dir,"reconstruction_error.txt")
        
        # Predict and store errors.
        predict_and_store_errors(new_model, file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features=TOTAL_FEATURES, n_files_per_batch=5, output_file=txt_file_path)

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