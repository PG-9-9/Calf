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
    
def sliding_window_with_positions(audio, window_size, step_size, sample_rate):
    num_samples_per_window = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    windows = []
    start_positions = []

    for start in range(0, len(audio) - num_samples_per_window + 1, step_samples):
        window = audio[start:start + num_samples_per_window]
        windows.append(window)
        start_positions.append(start)  # Track the start position of each window

    return windows, start_positions

def test_audio_data_generator(file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features, n_files_per_batch=30):
    concatenated_audio = np.array([])
    concatenated_file_tags = []  # To hold the filenames for each part of the concatenated audio
    file_start_positions = [0]  # Tracks start positions of each file within concatenated_audio

    for batch_index, file_path in enumerate(file_paths):
        # Load and append the current audio file to the concatenated stream
        audio = load_audio_file(file_path, sample_rate)
        concatenated_audio = np.concatenate((concatenated_audio, audio))
        # Update file start positions (end of this file is the start of the next)
        file_start_positions.append(len(concatenated_audio))
        concatenated_file_tags.append(file_path.split('/')[-1])  # Extract and store filename

        # Process the batch once enough files are loaded or on the last file
        if (batch_index + 1) % n_files_per_batch == 0 or batch_index == len(file_paths) - 1:
            # Generate windows and their start positions
            windows, window_start_positions = sliding_window_with_positions(concatenated_audio, window_size, step_size, sample_rate)
            sequences = [windows[i:i + expected_timesteps] for i in range(len(windows) - expected_timesteps + 1)]

            for i, sequence in enumerate(sequences):
                # Use window start positions to determine the sequence's originating files
                sequence_start = window_start_positions[i]
                sequence_end = window_start_positions[i] + len(sequence[0]) * expected_timesteps  # Approx end position
                
                # Identify which files this sequence spans
                sequence_files = []
                for j, file_start in enumerate(file_start_positions[:-1]):  # Exclude the last position as it's beyond the last file
                    file_end = file_start_positions[j + 1]
                    if sequence_start < file_end and sequence_end >= file_start:
                        sequence_files.append(concatenated_file_tags[j])

                features_sequence = np.array([extract_features(window, sample_rate) for window in sequence]).reshape(1, expected_timesteps, total_features)
                file_tag = ' + '.join(sequence_files)  # Combine filenames for sequences spanning multiple files
                yield features_sequence, file_tag

            # Reset for the next batch
            concatenated_audio = np.array([])
            concatenated_file_tags = []
            file_start_positions = [0]

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

def experiment_with_configurations(root_path, paths, sample_rate, evaluation_directory, hyperparameters_combinations):
    abnormal_path = paths['abnormal']  #  the abnormal data
    normal_path=paths['normal']
    validation_path=paths['validation']
    validation_files = [os.path.join(paths['validation'], f) for f in os.listdir(paths['validation']) if f.endswith('.wav')]
    print(f"Number of validation files: {len(validation_files)}")

    # List of file paths in abnormal 
    file_paths = [os.path.join(abnormal_path, f) for f in os.listdir(abnormal_path) if f.endswith('.wav')]
    
    for combination in hyperparameters_combinations:
        # Value extraction
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
        val_dataset = create_tf_dataset(
            {'validation': validation_path}, sample_rate, window_size, step_size, expected_timesteps, TOTAL_FEATURES, batch_size # Batch_size
        )
        #   Build and train the model
        autoencoder = build_autoencoder(expected_timesteps, TOTAL_FEATURES, lstm_neurons,   batch_size)
        autoencoder.fit(train_dataset, epochs=epochs, callbacks=[EarlyStopping(monitor='loss', patience=3),ResetStatesCallback()],validation_data=val_dataset)
        
        #   Save the model
        model_path = os.path.join(model_save_dir, "autoencoder_model.h5")
        autoencoder.save(model_path)
        logging.info(f"Model saved to {model_path}")
        
        # model_path='/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3/ws7_ss3.5_et10_lstm128_1epochs_bs32/autoencoder_model.h5'
        # Validate the model
        evaluate_model_and_plot_errors(autoencoder,val_dataset,model_save_dir)
        
        #   Rebuild for testing.        
        new_model=rebuild_model_for_prediction(expected_timesteps,TOTAL_FEATURES,lstm_neurons)
        new_model.load_weights(model_path)
        txt_file_path=os.path.join(model_save_dir,"reconstruction_error.txt")
        
        # Predict and store errors.
        predict_and_store_errors(new_model, file_paths, sample_rate, window_size, step_size, expected_timesteps, total_features=TOTAL_FEATURES, n_files_per_batch=30, output_file=txt_file_path)
        
        # Plot the RMS plots.
        data = parse_txt_file(txt_file_path)
        plot_rmse_by_time(data, model_save_dir) 

def main(evaluation_directory):
    root_path = 'Calf_Detection/Audio/Audio_Work_AE'
    paths = {
        'normal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_training_set',
        'abnormal': '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_subset',
        'validation':'/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_validation_set'
    }
    experiment_with_configurations(root_path, paths, SAMPLE_RATE, evaluation_directory, hyperparameters_combinations)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3'
    main(evaluation_directory)