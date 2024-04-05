import xml.etree.ElementTree as ET
import json
import os

def convert_xml_to_json(xml_file):
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
        "depth": depth
    }

    # Extract annotation-specific information from XML
    annotations = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        pose = obj.find("pose").text
        truncated = int(obj.find("truncated").text)
        difficult = int(obj.find("difficult").text)
        
        # Map the object name to a category_id.
        category_id_map = {"kalb": 1,"person":2}
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
            "area": area,
            "bbox": [xmin, ymin, xmax, ymax],
            "category_id": category_id,
            "id": len(annotations),
            "image_id": len(annotations),
            "name": name,
            "pose": pose,
            "truncated": truncated,
            "difficult": difficult
        }
        annotations.append(annotation_data)
        
    return annotations, image_data

# Directory containing the XML files
xml_dir = 'Calf_Detection/images/annotations'

# Directory to check for the matching filenames
check_dir = 'Calf_Detection/efficient_det/data/train'

# Get all filenames from the check_dir (without extensions)
check_files = {os.path.splitext(file)[0] for file in os.listdir(check_dir)}
print(check_files)


all_annotations = []
all_images = []
for file in os.listdir(xml_dir):
    if file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, file)
        annotations, image_data = convert_xml_to_json(xml_path)
        
        # Only include the data if the filename (without extension) exists in check_files
        if os.path.splitext(image_data["file_name"])[0] in check_files:
            print(os.path.splitext(image_data["file_name"])[0])
            all_annotations.extend(annotations)
            all_images.append(image_data)
            

# Prepare final JSON structure
data = {
    "annotations": all_annotations,
    "images": all_images
}

# Save to JSON file
json_file = 'Calf_Detection/output_annotations_v4.json'
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Annotations and image metadata saved to {json_file}")

# Checking if the number of images and number of annotations match
print(f"Annotations:{len(all_annotations)}")
