import os
import random
import shutil

# TODO:
# 1. Add points to manually adjust the split.
# 2. Remove existing images


def empty_directory(directory_path):
    for root, _, files in os.walk(directory_path, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)

        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)

    print(f"Directory '{directory_path}' has been emptied.")

def split_images(input_directory, train_directory, test_directory, val_directory, split_ratio=(0.7, 0.2, 0.1)):
    if abs(sum(split_ratio)-1.0) >1e-5:
        raise ValueError("The sum of split_ratio values should be equal to 1.0.")

    # Checking if the directories exist
    for folder in [train_directory, test_directory, val_directory]:
        os.makedirs(folder, exist_ok=True)

    # Get the list of image files
    image_files = [file for file in os.listdir(input_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    num_images = len(image_files)

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate the number of images for each split
    train_split = int(split_ratio[0] * num_images)
    test_split = int(split_ratio[1] * num_images)
    val_split = int(split_ratio[2] * num_images)

    # Move images to respective directories
    for idx, image_file in enumerate(image_files):
        src_path = os.path.join(input_directory, image_file)
        if idx < train_split:
            dest_path = os.path.join(train_directory, image_file)
        elif idx < train_split + test_split:
            dest_path = os.path.join(test_directory, image_file)
        else:
            dest_path = os.path.join(val_directory, image_file)

        shutil.copy(src_path, dest_path)

    print("Images distributed successfully.")

def count_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    file_count = sum(1 for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file)))
    print(f"Number of files in '{directory_path}': {file_count}")

input_directory = "Calf_Detection/efficient_det/data/img"
train_directory = "Calf_Detection/efficient_det/data/train"
test_directory = "Calf_Detection/efficient_det/data/test"
val_directory = "Calf_Detection/efficient_det/data/val"

split_ratio = (0.7, 0.2, 0.1)  # 70% for training, 20% for testing, and 10% for validation

directory_to_empty=[]
directory_to_empty.append(train_directory)
directory_to_empty.append(test_directory)
directory_to_empty.append(val_directory)
number_of_images=directory_to_empty


# Empty files  in each directory
for each_dir in directory_to_empty:
    directory_to_empty = each_dir
    print(directory_to_empty)
    empty_directory(directory_to_empty)
    count_files_in_directory(directory_to_empty)

# Split images
split_images(input_directory, train_directory, test_directory, val_directory, split_ratio)

# Count number of images after splitting:
print("Number of images after splitting")
for each_images in number_of_images:
    number_of_images=each_images
    count_files_in_directory(number_of_images)
