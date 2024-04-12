import zipfile
import os

def unzip_file(zip_path, output_dir):
    """
    Unzip a file to a specified output directory.

    Args:
    zip_path (str): The path to the zip file to be extracted.
    output_dir (str): The directory where the files will be extracted to.

    Returns:
    None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        print(f"All files extracted to {output_dir}")

def main(zip_path, output_dir):
    """
    Main function to unzip a file.

    Args:
    zip_path (str): The path to the zip file to be extracted. Defaults to 'path/to/your/zipfile.zip'.
    output_dir (str): The directory where the files will be extracted to. Defaults to 'path/to/output/directory'.

    Returns:
    None
    """
    unzip_file(zip_path, output_dir)

if __name__ == "__main__":
    # Call main function without arguments
    main()

    main('your/zip/file/path.zip', '/home/woody/iwso/iwso122h/Calf_Detection/Audio/Audio_Work_AE/abnormal_muliple_new')
