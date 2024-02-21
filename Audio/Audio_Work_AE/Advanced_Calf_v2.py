import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
import pickle

def load_audio_files(path, label, file_percentage):
    audio_files = []
    labels = []
    all_filenames = os.listdir(path)
    # random.shuffle(all_filenames)
    num_files_to_load = int(len(all_filenames) * file_percentage)

    for filename in all_filenames[:num_files_to_load]:
        if filename.endswith('.wav'):
            file_path = os.path.join(path, filename)
            audio, sample_rate = librosa.load(file_path, sr=None)
            audio_files.append(audio)
            labels.append(label)
    return audio_files, labels, sample_rate

def extract_features(audio_data, sample_rate, n_mfcc=13):
    features = []
    for audio in audio_data:
        # Basic Features: MFCCs, Spectral, Temporal
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
        spectral_features = np.array([np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)])

        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        autocorrelation = librosa.autocorrelate(audio)
        temporal_features = np.array([np.mean(zero_crossing_rate), np.mean(autocorrelation)])

        # Additional Features
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spec_flatness = librosa.feature.spectral_flatness(y=audio)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        rms = librosa.feature.rms(y=audio)
        additional_features = np.array([np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)])

        # Combine all features
        all_features = np.concatenate([mfccs_processed, spectral_features, temporal_features, additional_features])
        features.append(all_features)
    return np.array(features)

def enhanced_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
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
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# Enhanced autoencoder with LSTM
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

def train_and_evaluate_model(normal_path, abnormal_path, normal_percentage, abnormal_percentage, model_type, result_path):    
    # Load the datasets with specified file percentages for normal and abnormal paths
    abnormal_audio, _, _ = load_audio_files(abnormal_path, label=1, file_percentage=abnormal_percentage)
    normal_audio, _, sample_rate = load_audio_files(normal_path, label=0, file_percentage=normal_percentage)

    # Extract features
    normal_features = extract_features(normal_audio, sample_rate)
    abnormal_features = extract_features(abnormal_audio, sample_rate)

    # Split and scale the data
    X_train, X_val = train_test_split(normal_features, test_size=0.2, random_state=42)
    X_test = abnormal_features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_combined_test = np.concatenate((X_val_scaled, X_test_scaled))
    y_combined_test = np.concatenate((np.zeros(len(X_val_scaled)), np.ones(len(X_test_scaled))))

    # Model Training
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    if model_type == 'lstm':
        # Reshape data for LSTM
        timesteps = 1  # Need to finetune TODO
        n_features = X_train_scaled.shape[1]
        X_train_reshaped = X_train_scaled.reshape((-1, timesteps, n_features))
        X_val_reshaped = X_val_scaled.reshape((-1, timesteps, n_features))
        X_combined_test_reshaped = X_combined_test.reshape((-1, timesteps, n_features))
        
        autoencoder = enhanced_autoencoder_with_lstm(n_features, timesteps, n_features)
        print("**********")
        print(X_train_reshaped.shape)
        print(X_val_reshaped.shape)
        autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=400, batch_size=256, shuffle=True, validation_data=(X_val_reshaped, X_val_reshaped), callbacks=[early_stopping], verbose=0)

        # Model Evaluation
        reconstructed_combined = autoencoder.predict(X_combined_test_reshaped)
        mse_combined = np.mean(np.power(X_combined_test_reshaped - reconstructed_combined, 2), axis=1)
    else:
        autoencoder = enhanced_autoencoder(X_train_scaled.shape[1])
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=400, batch_size=256, shuffle=True, validation_data=(X_val_scaled, X_val_scaled), callbacks=[early_stopping], verbose=0)

        # Model Evaluation
        reconstructed_combined = autoencoder.predict(X_combined_test)
        mse_combined = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=1)

    precisions, recalls, thresholds = precision_recall_curve(y_combined_test, mse_combined)
    optimal_idx = np.argmax(2 * (precisions * recalls) / (precisions + recalls))
    optimal_threshold = thresholds[optimal_idx]
    optimal_predictions = (mse_combined > optimal_threshold).astype(int)
    optimal_cm = confusion_matrix(y_combined_test, optimal_predictions)

    # Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(optimal_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(result_path, f'confusion_matrix_{model_type}.png'))

    # Save the MSE curve
    mse_individual = np.mean(np.power(X_combined_test - reconstructed_combined, 2), axis=1)

    # Plot MSE for individual test files
    plt.figure(figsize=(10, 6))
    plt.plot(mse_individual, marker='o', linestyle='')
    plt.title('MSE for Individual Test Files')
    plt.xlabel('File Index')
    plt.ylabel('Mean Squared Error')
    mse_plot_path = os.path.join(result_path, f'mse_individual_{model_type}.png')
    plt.savefig(mse_plot_path)

    # Save Model
    model_save_path = os.path.join(result_path, f'autoencoder_{model_type}.h5')
    save_model(autoencoder, model_save_path)


if __name__ == "__main__":
    normal_path = "Calf_Detection/Audio/Audio_Work_AE/normal_calf_superset"  
    abnormal_path = "Calf_Detection/Audio/Audio_Work_AE/abnormal_calf_superset" 
    result_path = "Calf_Detection/Audio/Audio_Work_AE/Save_results"  

    normal_percentage = 0.01  # Percentage of files to process from normal path
    abnormal_percentage = 1.0  # Percentage of files to process from abnormal path

    # Train and evaluate using standard autoencoder
    train_and_evaluate_model(normal_path, abnormal_path, normal_percentage, abnormal_percentage, model_type='standard', result_path=result_path)

    # Train and evaluate using LSTM autoencoder
    # train_and_evaluate_model(normal_path, abnormal_path, normal_percentage, abnormal_percentage, model_type='lstm', result_path=result_path)