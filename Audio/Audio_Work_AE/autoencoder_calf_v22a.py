import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import librosa
import os
import logging
import matplotlib.dates as mdates
from datetime import datetime
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
SAMPLE_RATE = 44100
TOTAL_FEATURES = 23


# 5 * 25 = 125,   
hyperparameters_combinations = [
    {"window_size": 5, "step_size": 2.5, "expected_timesteps": 22, "lstm_neurons": 128, "epochs": 20, "batch_size": 32},
    {"window_size": 10, "step_size": 5, "expected_timesteps": 10, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
    {"window_size": 15, "step_size": 7.5, "expected_timesteps": 6, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
]

def create_model_directory(root_path, config):
    model_dir = os.path.join(root_path, "model_{}".format("_".join(map(str, config))))
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

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
            all_file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
            
            for i in range(0, len(all_file_paths), n_files_per_batch):
                batch_file_paths = all_file_paths[i:i+n_files_per_batch]
                
                for file_path in batch_file_paths:
                    audio = load_audio_file(file_path, sample_rate)
                    windows = sliding_window(audio, window_size, step_size, sample_rate)
                    
                    # Convert list of windows into sequences
                    sequences = [windows[j:j+expected_timesteps] for j in range(0, len(windows)-expected_timesteps+1, expected_timesteps)]
                    
                    for sequence in sequences:
                        if len(sequence) == expected_timesteps:
                            features_sequence = np.array([extract_features(window, sample_rate) for window in sequence])
                            yield features_sequence, features_sequence  # Yield each sequence of features
    
    return generator

def create_tf_dataset(paths, sample_rate, window_size, step_size, expected_timesteps, total_features, batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        generator=audio_data_generator(paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch=30),
        output_signature=(
            tf.TensorSpec(shape=(expected_timesteps, total_features), dtype=tf.float32),
            tf.TensorSpec(shape=(expected_timesteps, total_features), dtype=tf.float32),
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def save_features_in_batches(paths, sample_rate, combination, output_dir, n_files_per_batch):
    batch_size = combination["batch_size"]
    window_size = combination["window_size"]
    step_size = combination["step_size"]
    expected_timesteps = combination["expected_timesteps"]
    lstm_neurons = combination["lstm_neurons"]
    epochs = combination["epochs"]
    batch_size = combination["batch_size"]

    dataset_dirname = f"ws{window_size}_ss{step_size}_et{expected_timesteps}_lstm{lstm_neurons}_{epochs}epochs_bs{batch_size}"
    feature_save_dir = os.path.join(output_dir, dataset_dirname)
    os.makedirs(feature_save_dir, exist_ok=True)

    batch_counter = 0
    current_batch_features = []

    for path in paths.values():  # 
        file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wav')]
        for file_path in file_paths:
            audio = load_audio_file(file_path, sample_rate)
            windows = sliding_window(audio, window_size, step_size, sample_rate)
            
            for start in range(0, len(windows) - expected_timesteps + 1, expected_timesteps):
                sequence = windows[start:start + expected_timesteps]
                if len(sequence) == expected_timesteps:
                    sequence_features = np.array([extract_features(window, sample_rate) for window in sequence])
                    current_batch_features.append(sequence_features)
                    
                    if len(current_batch_features) >= batch_size:
                        batch_features = np.array(current_batch_features[:batch_size])
                        # For autoencoders, input data is the target data
                        np.savez_compressed(os.path.join(feature_save_dir, f'batch_{batch_counter}.npz'), features=batch_features, labels=batch_features)
                        current_batch_features = current_batch_features[batch_size:]
                        batch_counter += 1

    # Save any remaining features not forming a complete batch
    if current_batch_features:
        batch_features = np.array(current_batch_features)
        np.savez_compressed(os.path.join(feature_save_dir, f'batch_{batch_counter}.npz'), features=batch_features, labels=batch_features)

def npz_batch_generator(feature_dir, batch_size):
    npz_files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith('.npz')]
    npz_files.sort() 

    for npz_file in npz_files:
        data = np.load(npz_file)
        features = data['features']
        # (batch_size, expected_timesteps, total_features)
        yield features, features  # Yields features : both input and target 

def create_dataset_from_npz(feature_dir, expected_timesteps, total_features):
    def generator():
        for npz_file in sorted(os.listdir(feature_dir)):
            if npz_file.endswith('.npz'):
                data = np.load(os.path.join(feature_dir, npz_file))
                features = data['features']
                labels = data['labels']
                yield features, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, expected_timesteps, total_features], [None, expected_timesteps, total_features])
    ).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset



def build_autoencoder(expected_timesteps, total_features, lstm_neurons):
    input_layer = Input(shape=(expected_timesteps, total_features))

    # Encoder
    x = Conv1D(filters=32, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = LSTM(lstm_neurons, return_sequences=False)(x)  # TODO : Work on Statefullness

    # Replicate encoder output for decoder input
    x = RepeatVector(expected_timesteps)(x)

    # Decoder
    x = LSTM(lstm_neurons, return_sequences=True)(x)
    x = TimeDistributed(Dense(total_features))(x)

    autoencoder = Model(inputs=input_layer, outputs=x)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def verify_saved_data_shape(feature_dir):
    sample_npz_file = next((f for f in os.listdir(feature_dir) if f.endswith('.npz')), None)
    if sample_npz_file:
        data = np.load(os.path.join(feature_dir, sample_npz_file))
        features = data['features']
        labels = data['labels'] 
        print("Features shape:", features.shape)
        print("Labels shape:", labels.shape)
    else:
        print("No .npz files found in the directory.")

def rebuild_model_for_prediction(expected_timesteps, total_features, lstm_neurons):
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
    
def predict_and_store_errors(model, file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch, output_file):
    generator = test_audio_data_generator(file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch)
    
    with open(output_file, 'w') as f:
        for sequence, file_tag in generator:
            predicted_sequence = model.predict(sequence)
            error = np.mean(np.power(sequence - predicted_sequence, 2))
            f.write(f"{file_tag}: {error}\n")

def parse_txt_file(txt_file_path):
    data = []
    with open(txt_file_path, 'r') as f:
        for line in f:
            parts = line.split(': ')
            filenames = parts[0].replace("output_", "").split(' + ')
            rmse = float(parts[1])
            for filename in filenames:
                # Extract datetime from filename
                datetime_str = filename.replace(".wav", "")
                datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d_%H-%M-%S")
                data.append({'datetime': datetime_obj, 'rmse': rmse})
    return pd.DataFrame(data)

def plot_rmse_by_time(data, output_dir):
    # Ensure output directories exist
    granularities = ['hour', 'minute', 'window']
    for granularity in granularities:
        os.makedirs(os.path.join(output_dir, granularity), exist_ok=True)
    
    # Sort data
    data.sort_values(by='datetime', inplace=True)

    # Group data by day for minute and window granularity
    if 'minute' in granularities or 'window' in granularities:
        for date, date_group in data.groupby(data['datetime'].dt.date):
            if 'minute' in granularities:
                plot_minute_granularity(date_group, output_dir)
            if 'window' in granularities:
                plot_window_granularity(date_group, output_dir, date)
    
    # Group data by hour for hour granularity
    if 'hour' in granularities:
        for hour, hour_group in data.groupby(data['datetime'].dt.hour):
            plot_hour_granularity(hour_group, output_dir)

def plot_hour_granularity(data, output_dir):
    # Group data by day
    for date, group in data.groupby(data['datetime'].dt.date):
        plt.figure(figsize=(12, 6))
        
        # For plotting, ensure we cover all hours even if some are missing in the data
        group.set_index('datetime', inplace=True)
        group = group.resample('H').mean()  # Resample to hourly and take mean to ensure all hours are covered
        
        plt.plot(group.index.hour, group['rmse'], marker='o', linestyle='-', label=str(date))
        
        plt.title(f'RMSE for Each Hour on {date}')
        plt.xlabel('Hour of the Day')
        plt.ylabel('RMSE')
        plt.xticks(range(24))  # Ensure x-axis covers all 24 hours
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        day_output_dir = os.path.join(output_dir, 'hour')
        os.makedirs(day_output_dir, exist_ok=True)  # Ensure the directory exists
        plt.savefig(os.path.join(day_output_dir, f"{date}.png"))
        plt.close()

def plot_minute_granularity(data, output_dir):
    # Plot for each day, all minutes within the day
    for date, group in data.groupby(data['datetime'].dt.date):
        plt.figure(figsize=(10, 6))
        plt.plot(group['datetime'], group['rmse'], marker='o', linestyle='-', label=str(date))
        plt.title(f'RMSE per Minute for {date}')
        plt.xlabel('Time')
        plt.ylabel('RMSE')
        # Omit x-axis labels to avoid clutter
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'minute', f"{date}.png"))
        plt.close()

def plot_window_granularity(data, output_dir, date):
    plt.figure(figsize=(10, 6))
    plt.plot(data['datetime'], data['rmse'], marker='o', linestyle='-', label=str(date))
    plt.title(f'RMSE for Windows on {date}')
    plt.xlabel('Datetime')
    plt.ylabel('RMSE')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'window', f"{date}.png"))
    plt.close()

def evaluate_model_and_plot_errors(model, dataset, model_save_dir):
    all_reconstruction_errors = []
    sequence_indices = []  # To keep track of the sequence index

    for i, batch in enumerate(dataset):
        X_batch, _ = batch 
        reconstructed_batch = model.predict(X_batch)
        # Calculate MSE for each example in the batch
        batch_errors = np.mean(np.square(X_batch - reconstructed_batch), axis=(1, 2))
        all_reconstruction_errors.extend(batch_errors)
        sequence_indices.extend([i for i in range(len(batch_errors))])

    if len(all_reconstruction_errors) > 0:
        # Plot the reconstruction errors for each sequence
        plt.figure(figsize=(10, 6))
        plt.plot(sequence_indices, all_reconstruction_errors, marker='o', linestyle='-', color='blue')
        plt.title('Reconstruction Error for Each Sequence in Validation Set')
        plt.xlabel('Sequence Index')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.grid(True)
        plt.savefig(os.path.join(model_save_dir, "validation_reconstruction_errors.png"))
        plt.close()
        print(f"Plot saved to {os.path.join(model_save_dir, 'validation_reconstruction_errors.png')}")
    else:
        print("No data was processed. Please check the validation dataset.")

def experiment_with_configurations(evaluation_directory, hyperparameters_combinations):
    for combination in hyperparameters_combinations:
        # Setup for the current hyperparameter combination
        window_size = combination["window_size"]
        step_size = combination["step_size"]
        expected_timesteps = combination["expected_timesteps"]
        lstm_neurons = combination["lstm_neurons"]
        epochs = combination["epochs"]
        batch_size = combination["batch_size"]

        # dataset_dirname = f"ws{window_size}_ss{step_size}_et{expected_timesteps}_lstm{lstm_neurons}_{epochs}epochs_bs{batch_size}"
        dataset_dirname = f"ws{window_size}_ss{step_size}_et{expected_timesteps}_bs{batch_size}"
        
        feature_dir = os.path.join(evaluation_directory,dataset_dirname)

        # Ensure the directory exists
        if not os.path.exists(feature_dir):
            logging.error(f"Feature directory does not exist: {feature_dir}")
            continue
        
        # Create the TensorFlow dataset from saved .npz files
        dataset = create_dataset_from_npz(feature_dir,expected_timesteps,TOTAL_FEATURES)
        verify_saved_data_shape(feature_dir)

        # Setup the model
        model = build_autoencoder(expected_timesteps, TOTAL_FEATURES, lstm_neurons)

        model.fit(dataset, epochs=combination['epochs'])

        # Save the trained model
        model_save_path = os.path.join(evaluation_directory, dataset_dirname, "autoencoder_model.h5")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

        # Reset states 
        # model.reset_states()


def main(evaluation_directory):
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    paths = {
        'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_calf_subset'
        # 'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_subset',
        # 'validation':'/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_validation_set'
    }
    # for combination in hyperparameters_combinations:
    #     save_features_in_batches(paths, SAMPLE_RATE, combination, evaluation_directory, n_files_per_batch=30)
    #     print(f"Saved features in batches for combination: {combination}")   

    experiment_with_configurations(evaluation_directory, hyperparameters_combinations)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v4'
    main(evaluation_directory)