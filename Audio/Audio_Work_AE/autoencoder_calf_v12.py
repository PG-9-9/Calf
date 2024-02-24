import datetime
import numpy as np
import os
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import logging
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("AutoEncoder last ran on: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

def generate_mel_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def load_and_convert_to_spectrogram(file_path, sample_rate=44100):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    spectrogram = generate_mel_spectrogram(audio, sample_rate)
    return spectrogram

def pad_spectrograms(spectrograms, desired_shape):
    padded_spectrograms = []
    for spectrogram in spectrograms:
        padded = np.zeros(desired_shape)
        min_shape = (min(desired_shape[0], spectrogram.shape[0]), min(desired_shape[1], spectrogram.shape[1]))
        padded[:min_shape[0], :min_shape[1]] = spectrogram[:min_shape[0], :min_shape[1]]
        padded_spectrograms.append(padded)
    return np.array(padded_spectrograms)

def find_max_shape(spectrograms):
    max_shape = (0, 0)
    for spectrogram in spectrograms:
        if spectrogram.shape[0] > max_shape[0]:
            max_shape = (spectrogram.shape[0], max_shape[1])
        if spectrogram.shape[1] > max_shape[1]:
            max_shape = (max_shape[0], spectrogram.shape[1])
    return max_shape

def load_data_from_directory(directory_path, label, sample_rate=44100, max_files=None):
    spectrograms = []
    labels = []
    file_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if fname.endswith('.wav')]
    if max_files:
        file_paths = file_paths[:max_files]
    for file_path in file_paths:
        try:
            spectrogram = load_and_convert_to_spectrogram(file_path, sample_rate)
            spectrograms.append(spectrogram)
            labels.append(label)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
    if not spectrograms:
        logging.warning(f"No valid spectrograms were loaded from {directory_path}. Check the files and their format.")
    
    max_shape = find_max_shape(spectrograms)
    spectrograms = pad_spectrograms(spectrograms, max_shape)
    
    return np.array(spectrograms), np.array(labels)

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def prepare_dataset(normal_path, abnormal_path, sample_rate=44100, max_files=None):
    normal_spectrograms, normal_labels = load_data_from_directory(normal_path, 0, sample_rate, max_files=2000)
    abnormal_spectrograms, abnormal_labels = load_data_from_directory(abnormal_path, 1, sample_rate, max_files)
    if normal_spectrograms.size == 0 or abnormal_spectrograms.size == 0:
        raise ValueError("Failed to load any data from the specified directories. Please check the paths and file formats.")
    X = np.concatenate((normal_spectrograms, abnormal_spectrograms), axis=0)
    y = np.concatenate((normal_labels, abnormal_labels), axis=0)
    X = np.expand_dims(X, -1)  # Reshape for CNN input
    return train_test_split(X, y, test_size=0.2, random_state=42)

def hyperparameter_tuning(root_path, evaluation_path, data_path, config_list, use_cnn=True):
    best_accuracy = 0
    best_f1 = 0
    best_recall = 0
    best_config = None

    results_file = os.path.join(evaluation_path, 'hyperparameter_tuning_results.csv')
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Window Size", "Step Size", "Epochs", "Batch Size", "Accuracy", "F1 Score", "Recall"])

        for config in config_list:
            window_size, step_size, epochs, batch_size = config
            logging.info(f"Config - Window: {window_size}s, Step: {step_size}s, Epochs: {epochs}, Batch Size: {batch_size}")

            X_train, X_test, y_train, y_test = prepare_dataset(data_path['normal'], data_path['abnormal'], sample_rate=44100)

            input_shape = X_train.shape[1:]  # (n_mels, time_steps, 1)
            num_classes = 2  # Normal and Abnormal

            if use_cnn:
                model = build_cnn_model(input_shape, num_classes)
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

                y_pred = np.argmax(model.predict(X_test), axis=1)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                writer.writerow([window_size, step_size, epochs, batch_size, accuracy, f1, recall])

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_f1 = f1
                    best_recall = recall
                    best_config = config

                logging.info(f"Accuracy: {accuracy}, F1 Score: {f1}, Recall: {recall}")

    logging.info(f"Best Config: {best_config}, Best Accuracy: {best_accuracy}, Best F1: {best_f1}, Best Recall: {best_recall}")

if __name__ == "__main__":
    data_path = {
        'normal': "Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset",
        'abnormal': "Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_subset"
    }
    root_path = "Calf_Detection/Audio/Audio_Work_AE"
    evaluation_path = "Calf_Detection/Audio/Audio_Work_AE/Evaluation"
    config_list = [
        (10, 5, 50, 32),
        (15, 7, 100, 64),
        (20, 10, 150, 32),
        (25, 12, 200, 64),
        (30, 15, 250, 32),
    ]

    hyperparameter_tuning(root_path, evaluation_path, data_path, config_list)
