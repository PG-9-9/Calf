import os
from datetime import datetime
import shutil

def copy_files_exclude_dates(source_directory, destination_directory, exclude_dates):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for filename in os.listdir(source_directory):
        if filename.startswith("output_") and filename.endswith(".wav"):
            file_date_str = filename.split("_")[1]  # Extract date part from filename
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()

            if file_date not in exclude_dates:
                source_file_path = os.path.join(source_directory, filename)
                destination_file_path = os.path.join(destination_directory, filename)
                shutil.copy(source_file_path, destination_file_path)

def main():
    source_directory_path = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work/wav_data"
    destination_directory_path = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/normal_training_set"
    dates_to_exclude = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in ["2023-10-09", "2023-10-11","2023-10-12","2023-10-17","2023-10-25","2023-10-31","2023-11-04"]]
    copy_files_exclude_dates(source_directory_path, destination_directory_path, dates_to_exclude)
    print("Files copied successfully excluding specified dates.")

if __name__ == "__main__":
    main()
