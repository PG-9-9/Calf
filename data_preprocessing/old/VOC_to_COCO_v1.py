import xml.etree.ElementTree as ET
import json

def convert_xml_to_json(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract information from XML
    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    annotations = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        pose = obj.find("pose").text
        truncated = int(obj.find("truncated").text)
        difficult = int(obj.find("difficult").text)

        
        # Map the object name to a category_id. 
        category_id_map = {"kalb": 1}  # Currently only one category
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
            "id": len(annotations),  # assuming unique id for each annotation
            "image_id": len(annotations),  
            "name": name,
            "pose": pose,
            "truncated": truncated,
            "difficult": difficult
        }
        annotations.append(annotation_data)

    # Prepare final JSON structure
    data = {
        "annotations": annotations
    }

    return data

xml_file = 'Calf_Detection/images/annotations/2018-11-12_005501.xml'
data = convert_xml_to_json(xml_file)

# Save to JSON file
json_file = 'Calf_Detection/output_annotations_v1.json'
with open(json_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Annotations saved to {json_file}")
