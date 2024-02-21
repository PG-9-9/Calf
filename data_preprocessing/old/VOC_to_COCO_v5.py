import xml.etree.ElementTree as ET
import json
import os

# Global IDs
global_image_id = 0
global_annotation_id = 0

def count_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    file_count = sum(1 for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file)))
    print(f"Number of files in '{directory_path}': {file_count}")

def convert_xml_to_json(xml_file):
    global global_image_id
    global global_annotation_id

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract image-specific information from XML
    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    depth = int(root.find("size/depth").text)

    # Save image data
    image_data = {
        "file_name": filename,
        "height": height,
        "width": width,
        "depth": depth,
        "id": global_image_id
    }

    # Extract annotation-specific information from XML
    annotations = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        pose = obj.find("pose").text
        truncated = int(obj.find("truncated").text)
        difficult = int(obj.find("difficult").text)
        
        # Map the object name to a category_id
        category_id_map = {"kalb": 1, "person": 2}
        category_id = category_id_map.get(name, 0)
        
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Calculate area of bounding box
        area = (xmax - xmin) * (ymax - ymin)
        
        # Prepare annotation data
        annotation_data = {
            "file_name": filename,
            "area": area,
            "bbox": [xmin, ymin, xmax, ymax],
            "category_id": category_id,
            "id": global_annotation_id,
            "image_id": global_image_id,
            "name": name,
            "pose": pose,
            "truncated": truncated,
            "difficult": difficult
        }

        annotations.append(annotation_data)
        global_annotation_id += 1

    global_image_id += 1
    return annotations, image_data

# Directory containing the XML files
xml_dir = 'Calf_Detection/images/annotations'

# Directory to check for the matching filenames
# Write for train, test, val.
check_dir = 'Calf_Detection/efficient_det/data/train'

# Get all filenames from the check_dir (without extensions)
check_files = [os.path.splitext(file)[0] for file in os.listdir(check_dir) if not file.endswith('.xml')]
print(f"Number of files in '{check_dir}': {len(check_files)}")

all_annotations = []
all_images = []

for file in os.listdir(xml_dir):
    if file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, file)
        annotations, image_data = convert_xml_to_json(xml_path)
        
        if os.path.splitext(image_data["file_name"])[0] in check_files:
            all_images.append(image_data)
            all_annotations.extend(annotations)
            check_files.remove(os.path.splitext(image_data["file_name"])[0])

# Now, check_files contains filenames for which annotations don't exist
# Let's delete these files
for file_without_annotation in check_files:
    os.remove(os.path.join(check_dir, file_without_annotation + ".jpg"))  

# Remove annotations if its corresponding image_object file_name doesn't exist
valid_image_file_names = {img['file_name'] for img in all_images}
all_annotations = [anno for anno in all_annotations if anno['file_name'] in valid_image_file_names]

# Prepare final JSON structure
image_filename_to_id = {img['file_name']: img['id'] for img in all_images}

# Filter annotations based on the images we've collected
filtered_annotations = [anno for anno in all_annotations if anno['file_name'] in image_filename_to_id]

# Update annotation image_ids based on the image_filename_to_id mapping
for anno in filtered_annotations:
    anno['image_id'] = image_filename_to_id[anno['file_name']]

# Prepare final JSON structure
# Adding the categories to the final JSON structure
category_data=[
            {
                "id":1,
                "name":"kalb",
                "supercategory":"cattle"
            },
            {
                "id":2,
                "name":"person",
                "supercategory":"cattle"
                }
            ]
data = {
    "annotations": filtered_annotations,
    "images": all_images,
    "categories": category_data
}

# Save to JSON file
json_file = 'Calf_Detection/efficient_det/data/train/train.json'
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Annotations and image metadata saved to {json_file}")

# Checking if the number of images and number of annotations match
print(f"Number of Annotations: {len(filtered_annotations)}")
print(f"Number of image objects: {len(all_images)}")
count_files_in_directory(check_dir)