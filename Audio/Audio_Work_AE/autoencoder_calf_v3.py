import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
#from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed, BatchNormalization, Dropout, Activation

normal_calf_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_training_set"
abnormal_calf_path="/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_subset"
SAMPLE_RATE=44100

def load_audio_files(path, label):
    audio_files = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            file_path = os.path.join(path, filename)
            audio, sample_rate = librosa.load(file_path, sr=None)
            audio_files.append(audio)
            labels.append(label)
    return audio_files, labels, sample_rate

def extract_mfccs(audio, sample_rate, n_mfcc=13):# MFCCs (Power Spectrum)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def extract_spectral_features(audio, sample_rate):# Spectral Features (spectral centroid, spectral roll-off, and spectral contrast):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

def extract_temporal_features(audio):# Temporal Features 
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

def augment_audio(audio, sample_rate, noise_factor=0.005, shift_max=0.2):# Data Augmentation:
    augmented_audio = audio
    noise = np.random.randn(len(augmented_audio))
    augmented_audio = augmented_audio + noise_factor * noise
    return augmented_audio

def extract_features(audio_data, sample_rate):
    features = []
    for audio in audio_data:
        mfccs = extract_mfccs(audio, sample_rate)
        spectral_features = extract_spectral_features(audio, sample_rate)
        temporal_features = extract_temporal_features(audio)
        all_features = np.concatenate([mfccs, spectral_features, temporal_features])
        features.append(all_features)
    return np.array(features)

def enhanced_autoencoder_with_lstm(input_dim, timesteps, n_features):
    input_layer = Input(shape=(timesteps, n_features))

    # Encoder with LSTM
    encoder = LSTM(128, activation='relu', return_sequences=True)(input_layer)
    encoder = LSTM(64, activation='relu', return_sequences=False)(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)

    # Repeat Vector
    repeat_vector = RepeatVector(timesteps)(encoder)

    # Decoder with LSTM
    decoder = LSTM(64, activation='relu', return_sequences=True)(repeat_vector)
    decoder = LSTM(128, activation='relu', return_sequences=True)(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.1)(decoder)
    output_layer = TimeDistributed(Dense(n_features, activation='sigmoid'))(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def main():
    # Load the datasets
    abnormal_audio, abnormal_labels, _ = load_audio_files(abnormal_calf_path, label=1)
    normal_audio, normal_labels, sample_rate = load_audio_files(normal_calf_path, label=0)

    # Extract features for both normal and abnormal data
    normal_features = extract_features(normal_audio, SAMPLE_RATE)
    abnormal_features = extract_features(abnormal_audio, SAMPLE_RATE)

    # Data preprocessing steps
    X_train, X_val = train_test_split(normal_features, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Determine suitable timesteps for LSTM
    timesteps = 1  # Example timestep, adjust based on your dataset
    n_features = normal_features.shape[1]
    X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
    X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))

    # Train the autoencoder
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder = enhanced_autoencoder_with_lstm(input_dim=n_features, timesteps=timesteps, n_features=n_features)
    autoencoder.fit(
        X_train_reshaped, X_train_reshaped,
        epochs=400,
        batch_size=256,
        validation_data=(X_val_reshaped, X_val_reshaped),
        callbacks=[early_stopping],
        verbose=1
    )
    autoencoder.save('Encoder_Model.keras')

if __name__ == "__main__":
    main()