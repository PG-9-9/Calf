import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import librosa
import os
import psutil
import time
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, RepeatVector, TimeDistributed, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Initialize logging
SAMPLE_RATE = 44100  # Hz
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting feature extraction and model training: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

def sliding_window(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []
    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
    return windows

def generate_spectrogram_for_window(window, sample_rate=44100, n_mels=128):
    spectrogram = librosa.feature.melspectrogram(y=window, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=n_mels)
    S_DB = librosa.power_to_db(spectrogram, ref=np.max)
    return S_DB

# CNN Feature Extractor Function
def build_cnn_feature_extractor(input_shape=(128, 128, 1)):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM AutoEncoder Function
def simplified_autoencoder_with_lstm(timesteps, n_features, lstm_neurons):
    input_layer = Input(shape=(timesteps, n_features))
    encoder = LSTM(lstm_neurons, activation='relu', return_sequences=False)(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)
    repeat_vector = RepeatVector(timesteps)(encoder)
    decoder = LSTM(lstm_neurons, activation='relu', return_sequences=True)(repeat_vector)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(0.1)(decoder)
    output_layer = TimeDistributed(Dense(n_features))(decoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def simplified_autoencoder(n_features):
    input_layer = Input(shape=(n_features,))
    encoder = Dense(128, activation='relu')(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(32, activation='relu')(encoder)
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

# Model Evaluation Function
def model_evaluation(autoencoder, X_test, y_test, evaluation_directory, model_type, features_per_file_abnormal):
    # Generate predictions for test set
    mse_test = []
    for features in X_test:  # Assuming X_test is already prepared similarly to features_per_file_abnormal
        reconstructed = autoencoder.predict(features)
        mse = np.mean(np.power(features - reconstructed, 2), axis=(1, 2))
        mse_test.extend(mse)
    mse_test = np.array(mse_test)
    
    # Determine optimal threshold from test set MSE
    precisions, recalls, thresholds = precision_recall_curve(y_test, mse_test)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Predict on abnormal files
    file_predictions = []
    for features in features_per_file_abnormal:
        reconstructed = autoencoder.predict(features)
        mse = np.mean(np.power(features - reconstructed, 2), axis=(1, 2))
        # Classify the file as abnormal if a significant number of its windows are above the threshold
        abnormal_windows = np.sum(mse > optimal_threshold)
        file_predictions.append(abnormal_windows > len(features) * 0.5)  # Example threshold: over 50% of windows are abnormal

    # Evaluate file-level predictions
    y_true_abnormal = np.ones(len(features_per_file_abnormal))  # All files in features_per_file_abnormal are abnormal
    accuracy = accuracy_score(y_true_abnormal, file_predictions)
    precision = precision_score(y_true_abnormal, file_predictions)
    recall = recall_score(y_true_abnormal, file_predictions)
    f1 = f1_score(y_true_abnormal, file_predictions)

    # Log the evaluation metrics
    logging.info(f"File-level Evaluation Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Optionally, visualize the results
    plt.figure(figsize=(10, 6))
    sns.histplot(mse_test, bins=50, kde=True)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--')
    plt.title('MSE Distribution of Test Windows with Optimal Threshold')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Density')
    plt.savefig(os.path.join(evaluation_directory, f'mse_distribution_{model_type}.png'))
    plt.close()

    # ROC Curve for window-level predictions
    fpr, tpr, _ = roc_curve(y_test, mse_test)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(evaluation_directory, f'roc_curve_{model_type}.png'))
    plt.close()
        
def extract_features_with_cnn(file_paths, cnn_model, sample_rate=44100):
    features = []
    for file_path in file_paths:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        spectrogram = generate_spectrogram_for_window(audio, sample_rate)
        spectrogram = librosa.util.fix_length(spectrogram, size=128)  # Ensure the spectrogram is exactly 128x128
        spectrogram = np.resize(spectrogram, (128, 128))
        spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
        feature = cnn_model.predict(np.array([spectrogram]))[0]
        features.append(feature)
    return np.array(features)

def load_data_from_directory(directory_path, cnn_model, sample_rate=44100, max_files=None, window_size=0.5, step_size=0.25):
    """
    Modified to return a list of feature arrays per audio file instead of flattening them.
    This allows us to track which features belong to which audio file.
    """
    file_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if fname.endswith('.wav')][:max_files]
    features_per_file = []

    for file_path in file_paths:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        windows = sliding_window(audio, window_size, step_size, sample_rate)
        features = []
        for window in windows:
            spectrogram = generate_spectrogram_for_window(window, sample_rate)
            spectrogram = np.resize(spectrogram, (128, 128))
            spectrogram = np.expand_dims(spectrogram, axis=-1)
            feature = cnn_model.predict(np.array([spectrogram]))[0]
            features.append(feature)
        features_per_file.append(np.array(features))

    return features_per_file

def hyperparameter_tuning(root_path, evaluation_path, data_path, config_list, sample_rate=44100):
    best_config = None
    best_accuracy = 0
    cnn_input_shape = (128, 128, 1)
    cnn_model = build_cnn_feature_extractor(cnn_input_shape)

    for config in config_list:
        window_size, step_size, lstm_neurons, epochs, batch_size = config
        logging.info(f"Testing config: Window={window_size}, Step={step_size}, LSTM={lstm_neurons}, Epochs={epochs}, Batch={batch_size}")

        normal_features, normal_labels = load_data_from_directory(data_path['normal'], cnn_model, sample_rate)
        abnormal_features, abnormal_labels = load_data_from_directory(data_path['abnormal'], cnn_model, sample_rate)
        
        features = np.concatenate((normal_features, abnormal_features), axis=0)
        labels = np.array([0] * len(normal_features) + [1] * len(abnormal_features))

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
        X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

        X_train_reshaped = X_train_scaled.reshape((-1, 1, X_train_scaled.shape[1]))
        X_test_reshaped = X_test_scaled.reshape((-1, 1, X_test_scaled.shape[1]))
        X_abnormal_prepared = [scaler.transform(features.reshape(features.shape[0], -1)).reshape((-1, 1, features.shape[1])) for features in abnormal_features]

        if lstm_neurons > 0:  # Assuming LSTM neurons > 0 indicates use of LSTM model
            autoencoder = simplified_autoencoder_with_lstm(1, X_train_reshaped.shape[2], lstm_neurons)
        else:
            autoencoder = simplified_autoencoder(X_train_reshaped.shape[2])

        autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=epochs, batch_size=batch_size, validation_data=(X_test_reshaped, X_test_reshaped), callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

        model_type_label = 'lstm' if lstm_neurons > 0 else 'non_lstm'
        model_evaluation(autoencoder, X_test_reshaped, y_test, evaluation_path, model_type_label, X_abnormal_prepared)


    logging.info(f"Best Config: {best_config}, Best ROC AUC: {best_accuracy}")

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

        ]
        

        root_path = "Calf_Detection/Audio/Audio_Work_AE"
        evaluation_path='Calf_Detection/Audio/Audio_Work_AE/New_Files'
        # Hyperparameter tuning for LSTM Autoencoder
        hyperparameter_tuning(root_path, evaluation_path, data_path, config_list)

        # Hyperparameter tuning for Dense Autoencoder
        # hyperparameter_tuning(root_path, evaluation_path , data_path, config_list, use_lstm=False)

    except Exception as e:
        logging.error("An error occurred in the main script", exc_info=True)