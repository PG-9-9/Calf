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
tf.config.run_functions_eagerly(True)

log_filename = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v2/processing_log.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename, 'a'), logging.StreamHandler()])

class ResetStatesCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()

# Constants
SAMPLE_RATE = 44100
TOTAL_FEATURES = 23

# 5 * 25 = 125,   
hyperparameters_combinations = [
    {"window_size": 5, "step_size": 2.5, "expected_timesteps": 23, "lstm_neurons": 128, "epochs": 20, "batch_size": 32}
    # {"window_size": 10, "step_size": 5, "expected_timesteps": 11, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
    # {"window_size": 15, "step_size": 7.5, "expected_timesteps": 7, "lstm_neurons": 64, "epochs": 20, "batch_size": 32},
]

def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

def npz_data_generator(npz_paths, batch_size=32, expected_timesteps=None, total_features=TOTAL_FEATURES):
    def generator():
        batch_features = []
        for npz_path in npz_paths:
            data = np.load(npz_path)
            features = data['features']
            
            # Pad features if they do not match the expected timesteps
            if expected_timesteps is not None and features.shape[1] != expected_timesteps:
                print("Dont match")
                difference = expected_timesteps - features.shape[1]
                padded_features = np.pad(features, ((0, 0), (0, difference), (0, 0)), 'constant')
                features = padded_features
            
            batch_features.append(features)
            if len(batch_features) == batch_size:
                yield np.stack(batch_features), np.stack(batch_features)
                batch_features = []
        
        # Handle the last batch by padding it to reach the expected batch size
        if batch_features:
            while len(batch_features) < batch_size:
                # Pad with zeros of shape (expected_timesteps, total_features)
                batch_features.append(np.zeros((expected_timesteps, total_features)))
            yield np.stack(batch_features), np.stack(batch_features)
    
    return generator

def create_tf_dataset_from_npz(npz_dir, batch_size=32, expected_timesteps=None, total_features=TOTAL_FEATURES):
    npz_paths = [os.path.join(npz_dir, filename) for filename in os.listdir(npz_dir) if filename.endswith('.npz')]
    dataset = tf.data.Dataset.from_generator(
        generator=npz_data_generator(npz_paths, batch_size, expected_timesteps, total_features),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, expected_timesteps, total_features), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, expected_timesteps, total_features), dtype=tf.float32)
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

def build_model(train_dataset, val_dataset, expected_timesteps, total_features, lstm_neurons, epochs, batch_size):  
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
    
    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    
    return model

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

def evaluate_validation(model, dataset, model_save_dir):
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
        logging.info(f"Plot saved to {os.path.join(model_save_dir, 'validation_reconstruction_errors.png')}")

    else:
        print("No data was processed. Please check the validation dataset.")
        logging.info("No data was processed. Please check the validation dataset to construct validation_reconstruction_errors.png.")

def identify_contributing_files(file_paths, window_start, window_end, window_size, step_size, sample_rate):
    # Convert window and step sizes to samples
    window_samples = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)

    # Calculate the total samples contributed by each file to identify their segments in the concatenated audio
    total_samples_per_file=60 * sample_rate
    file_samples = [total_samples_per_file for _ in file_paths]
    file_end_samples = np.cumsum(file_samples)

    # Determine the contributing files and their segments for the window
    contributing_files = []
    for i, file_path in enumerate(file_paths):
        file_start_sample = 0 if i == 0 else file_end_samples[i - 1]
        file_end_sample = file_end_samples[i]

        # Check if the window overlaps with the current file segment
        if window_end > file_start_sample and window_start < file_end_sample:
            # Calculate the overlap segment in the current file
            overlap_start = max(window_start - file_start_sample, 0) / sample_rate
            overlap_end = min(window_end - file_start_sample, file_end_sample - file_start_sample) / sample_rate
            contributing_files.append((file_path, (overlap_start, overlap_end)))

    return contributing_files

def evaluate_test(npz_dir, model, output_file_path, sample_rate, window_size, step_size):
    with open(output_file_path, 'w') as file:
        # Iterate over all npz files in the given directory
        for filename in os.listdir(npz_dir):
            if filename.endswith('.npz'):
                npz_path = os.path.join(npz_dir, filename)
                # Load the data with allow_pickle=True to load Python objects
                data = np.load(npz_path, allow_pickle=True)
                features = data['features']
                
                if 'metadata' in data:
                    # Load metadata, assuming it's saved as a dict
                    metadata = data['metadata'].item()
                else:
                    metadata = {"file_paths": "Unknown", "sequence_start": "Unknown", "sequence_end": "Unknown"}

                # Model prediction for the sequence
                predictions = model.predict(features[np.newaxis, :, :])

                # Calculate reconstruction error (MSE) for each window in the sequence
                reconstruction_errors = np.mean((features - predictions.squeeze()) ** 2, axis=1)  # axis=1 computes MSE for each window
                
                # Iterate through each window to identify contributing files and log details
                for i, error in enumerate(reconstruction_errors):
                    window_start_sample = metadata['sequence_start'] + i * step_size * sample_rate
                    window_end_sample = window_start_sample + window_size * sample_rate
                    contributing_files = identify_contributing_files(metadata['file_paths'], window_start_sample, window_end_sample, window_size, step_size, sample_rate)
                    
                    # Log the reconstruction errors along with contributing file details for each window
                    for fp, seg in contributing_files:
                        file_str = f"File: {fp}, Segment (start, end) in seconds: {seg}, Window {i}, Reconstruction Error: {error}\n"
                        print(file_str)  # Print to console for verification
                        file.write(file_str)  # Write to output file


def builder(normal_path,    abnormal_path,  validation_path, configs, evaluation_directory):
    window_size = configs["window_size"]
    step_size = configs["step_size"]
    expected_timesteps = configs["expected_timesteps"]
    lstm_neurons = configs["lstm_neurons"]
    epochs = configs["epochs"]
    batch_size = configs["batch_size"]  
    # abnormal_path=normal_path
    validation_path=normal_path
    
    
    # Creating the directories for storing the files.
    dataset_dirname = f"ws{window_size}_ss{step_size}_et{expected_timesteps}_lstm{lstm_neurons}_{epochs}epochs_bs{batch_size}"
    model_save_dir = os.path.join(evaluation_directory, dataset_dirname)
    model_save_file = os.path.join(model_save_dir, "autoencoder_model.h5")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Creating tf_dataset from the npz features
    train_dataset = create_tf_dataset_from_npz(normal_path, batch_size, expected_timesteps, TOTAL_FEATURES)
    val_dataset = create_tf_dataset_from_npz(validation_path,   batch_size,    expected_timesteps, TOTAL_FEATURES)
    
    # Training the model and saving it
    autoencoder=build_model(train_dataset, val_dataset, expected_timesteps, TOTAL_FEATURES, lstm_neurons, epochs, batch_size)
    autoencoder.fit(train_dataset, epochs=epochs, validation_data=val_dataset,  callbacks=[EarlyStopping(monitor='loss', patience=3), ResetStatesCallback()])
    autoencoder.save(model_save_file)
    logging.info(f"Model saved to {model_save_file}")
    
    # Evaluate the validation set
    evaluate_validation(autoencoder, val_dataset, model_save_dir)
    
    # Rebuild the model for testing ( Statefullnes )
    autoencoder_test=rebuild_model_for_prediction(expected_timesteps,TOTAL_FEATURES,lstm_neurons)
    autoencoder_test.load_weights(model_save_file)
    
    # Predict and store errors.
    txt_file_path=os.path.join(model_save_dir,"reconstruction_error.txt")
    # evaluate_test(abnormal_path, autoencoder_test, txt_file_path)
    evaluate_test(abnormal_path, autoencoder_test, txt_file_path, SAMPLE_RATE, window_size, step_size)

    
def main(evaluation_directory):
    configs = hyperparameters_combinations[0]
    
    normal_path='/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v2/ws5_ss2.5_et23_lstm128_20epochs_bs32/data'
    abnormal_path='/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v2/ws5_ss2.5_et23_lstm128_20epochs_bs32/test_data'
    validation_path='/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v2/ws5_ss2.5_et23_lstm128_20epochs_bs32/data'

    # Train, Validate and Test.
    builder(normal_path,    abnormal_path,  validation_path, configs, evaluation_directory)

if __name__ == '__main__':
    evaluation_directory = '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/View_Files/Debug_v3'
    main(evaluation_directory)