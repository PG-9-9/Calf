import os
import shutil
from datetime import datetime, timedelta

import os
import shutil
from datetime import datetime, timedelta

def find_closest_time(target_datetime, all_datetimes):
    """Find the datetime closest to the target in a list of datetimes."""
    return min(all_datetimes, key=lambda x: abs(x - target_datetime))

def get_all_file_datetimes(source_dir):
    """Extract and return all datetime objects from file names in the source directory."""
    datetimes = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".wav"):
            file_datetime_str = '_'.join(filename.split('_')[1:]).rsplit('.', 1)[0]
            file_datetime = datetime.strptime(file_datetime_str, "%Y-%m-%d_%H-%M-%S")
            datetimes.append(file_datetime)
    return datetimes

def copy_wav_files_in_date_time_range(source_dir, dest_dir, specified_date_times):
    os.makedirs(dest_dir, exist_ok=True)
    
    all_file_datetimes = get_all_file_datetimes(source_dir)
    total_files_copied = 0

    for specified_date_time in specified_date_times:
        # Parse the specified date and time
        specified_datetime = datetime.strptime(specified_date_time, "%Y-%m-%d_%H-%M-%S")
        
        # Find the closest datetime to the specified_datetime
        closest_datetime = find_closest_time(specified_datetime, all_file_datetimes)
        
        # Calculate start and end datetime based on the closest datetime
        start_datetime = closest_datetime - timedelta(hours=2)
        end_datetime = closest_datetime + timedelta(hours=2)

        files_copied = 0

        for filename in os.listdir(source_dir):
            if filename.endswith(".wav"):
                file_datetime_str = '_'.join(filename.split('_')[1:]).rsplit('.', 1)[0]
                file_datetime = datetime.strptime(file_datetime_str, "%Y-%m-%d_%H-%M-%S")

                if start_datetime <= file_datetime <= end_datetime:
                    source_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(dest_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    files_copied += 1

        print(f"Files copied for {specified_date_time} (closest: {closest_datetime.strftime('%Y-%m-%d_%H-%M-%S')}): {files_copied}")
        total_files_copied += files_copied

    return total_files_copied
if __name__ == "__main__":
    source_directory = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_single_day/04_Nov"
    destination_directory = "/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_validation_subset"
    
    specified_date_times = [
        "2023-11-04_13-53-01"
    ]

    num_files_copied = copy_wav_files_in_date_time_range(source_directory, destination_directory, specified_date_times)
    print(f"Total number of files copied: {num_files_copied}")
