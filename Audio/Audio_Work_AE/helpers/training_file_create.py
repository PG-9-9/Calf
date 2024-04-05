import os
from datetime import datetime
import shutil

def copy_files_with_dates(source_directory, destination_directory, dates):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for filename in os.listdir(source_directory):
        if filename.startswith("output_") and filename.endswith(".wav"):
            file_date_str = filename.split("_")[1]  # Extract date part from filename
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")

            if file_date.date() in dates:
                source_file_path = os.path.join(source_directory, filename)
                destination_file_path = os.path.join(destination_directory, filename)
                shutil.copy(source_file_path, destination_file_path)

def main():
    source_directory_path = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_set"
    destination_directory_path = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_single_day/04_Nov"
    dates_to_copy = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in ["2023-11-04"]]
    copy_files_with_dates(source_directory_path, destination_directory_path, dates_to_copy)
    print("Files copied successfully.")

if __name__ == "__main__":
    main()
