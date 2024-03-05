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
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SAMPLE_RATE = 44100
WINDOW_SIZE = 0.5  # Window size in seconds
STEP_SIZE = 0.25  # Step size in seconds
EXPECTED_TIMESTEPS = 10
TOTAL_FEATURES = 23

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

def audio_data_generator(paths, sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    def generator():
        for label, path in paths.items():
            for filename in os.listdir(path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(path, filename)
                    audio = load_audio_file(file_path, sample_rate)
                    windows = sliding_window(audio, window_size, step_size, sample_rate)
                    if len(windows) >= EXPECTED_TIMESTEPS:
                        features = [extract_features(window, sample_rate) for window in windows[:EXPECTED_TIMESTEPS]]
                        features = np.stack(features)  # This should shape the features as (EXPECTED_TIMESTEPS, TOTAL_FEATURES)
                        yield features, features
    return generator

def create_tf_dataset(paths, sample_rate, window_size, step_size, batch_size=32):
    output_signature = (
        tf.TensorSpec(shape=(None, TOTAL_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(None, TOTAL_FEATURES), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        generator=audio_data_generator(paths, sample_rate, window_size, step_size),
        output_signature=output_signature
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def build_autoencoder(timesteps=EXPECTED_TIMESTEPS, features=TOTAL_FEATURES):  # Adjust features to match the total feature count
    input_shape = (timesteps, features)
    input_layer = Input(shape=input_shape)
    # Encoder
    x = LSTM(128, activation='relu', return_sequences=False)(input_layer)
    x = RepeatVector(timesteps)(x)
    # Decoder
    x = LSTM(128, activation='relu', return_sequences=True)(x)
    output_layer = TimeDistributed(Dense(features))(x)
    autoencoder = models.Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def main():
    root_path = 'your/dataset/path'
    paths = {
        'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset',
        'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset'
    }

    # Create TensorFlow dataset
    dataset = create_tf_dataset(paths, SAMPLE_RATE, WINDOW_SIZE, STEP_SIZE)
    for features, labels in dataset.take(1):
        print(features.shape, labels.shape)

    # Build the model
    autoencoder = build_autoencoder(timesteps=EXPECTED_TIMESTEPS, features=TOTAL_FEATURES)  

    # Train the model
    autoencoder.fit(dataset, epochs=20, callbacks=[EarlyStopping(monitor='loss', patience=3)])

    # Save the model
    autoencoder.save('/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3/awesome.h5')
    
if __name__ == '__main__':
    main()
