import datetime
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import io
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
import tensorflow as tf
import pathlib
import itertools
import random
import subprocess
import xml.etree.ElementTree as ET
import glob

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

gpu_memory_fraction = 0.8
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(gpu_memory_fraction * 1024))]
            )
    except RuntimeError as e:
        print(e)

# for model_version in ["d0"]:
for model_version in ["d2", "d3", "d4", "d5", "d6", "d7"]:
    #Training 
    command = [
        'python3',
        'Calf_Detection/models/research/object_detection/model_main_tf2.py',
        '--pipeline_config_path=Calf_Detection/new_app/New_Models/config/efficientdet_{}_coco17_tpu-32.config'.format(model_version),
        '--model_dir=Calf_Detection/new_app/New_Models/training/epochs_10/efficientdet_{}_coco17_tpu-32'.format(model_version),
        '--alsologtostderr'
    ]
    subprocess.run(command, check=True)

    #Export
    command = [
        'python3',
        'Calf_Detection/models/research/object_detection/exporter_main_v2.py',
        '--trained_checkpoint_dir', 'Calf_Detection/new_app/New_Models/training/epochs_10/efficientdet_{}_coco17_tpu-32'.format(model_version),
        '--output_directory', 'Calf_Detection/new_app/finetuned/epochs_10/efficientdet_{}_coco17_tpu-32'.format(model_version),
        '--pipeline_config_path', 'Calf_Detection/new_app/New_Models/config/efficientdet_{}_coco17_tpu-32.config'.format(model_version)
    ]
    subprocess.run(command, check=True)

    # Evaluation
    # command = [
    #     'python3',
    #     'Calf_Detection/models/research/object_detection/model_main_tf2.py',
    #     '--pipeline_config_path', 'Calf_Detection/new_app/New_Models/config/efficientdet_{}_coco17_tpu-32.config'.format(model_version),
    #     '--model_dir', 'Calf_Detection/new_app/New_Models/training/epochs_10/efficientdet_{}_coco17_tpu-32'.format(model_version),
    #     '--checkpoint_dir', 'Calf_Detection/new_app/New_Models/training/epochs_10/efficientdet_{}_coco17_tpu-32'.format(model_version),
    #     '--alsologtostderr'
    # ]
    # subprocess.run(command, check=True)
    TEST_IMAGE_PATHS = glob.glob('Calf_Detection/new_app/test/*.jpg')

    # Metrics Initialization
    false_negative = 0
    true_negative = 0
    true_positive = 0
    false_positive = 0
    total_ground_truth = len(TEST_IMAGE_PATHS)
    low_iou_images = []

    def load_image_into_numpy_array(path):
        img_data = tf.io.gfile.GFile(path, 'rb').read()
        image = Image.open(BytesIO(img_data))
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def compute_iou(boxA, boxB):
        """Computes Intersection over Union (IoU) between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def extract_bbox_from_xml(xml_path):
        """Extracts bounding box information from an XML annotation file."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            bbox = member.find('bndbox')
            return [int(bbox.find(pos).text) for pos in ['xmin', 'ymin', 'xmax', 'ymax']]

    def get_ground_truth_for_image(image_path):
        """Gets ground truth bounding box for a given image."""
        xml_path = image_path.replace('.jpg', '.xml')  
        return extract_bbox_from_xml(xml_path)

    def load_model(pipeline_file, model_dir):
        """Loads the saved model from checkpoint."""
        configs = config_util.get_configs_from_pipeline_file(pipeline_file)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        
        # Restore checkpoint
        filenames = list(pathlib.Path(model_dir).glob('*.index'))
        filenames.sort()
        checkpoint_path = str(filenames[-1]).replace('.index','')
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(checkpoint_path).expect_partial()
        
        return detection_model

    def detect_objects(detection_model, image_np):
        """Detects objects in an image using the trained model."""
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)
        return detections

    @tf.function
    def detect_fn(image, model):
        """TF function for object detection."""
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)
        return detections
    
    detection_model = load_model('Calf_Detection/new_app/New_Models/config/efficientdet_{}_coco17_tpu-32.config'.format(model_version),
                                 'Calf_Detection/new_app/New_Models/training/epochs_10/efficientdet_{}_coco17_tpu-32'.format(model_version))

    thresholds = [0.25, 0.50, 0.75, 1.00]
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        # Reset the metrics for each threshold
        false_negative = 0
        true_negative = 0
        true_positive = 0
        false_positive = 0

        for image_path in TEST_IMAGE_PATHS:
            image_np = load_image_into_numpy_array(image_path)
            detections = detect_objects(detection_model, image_np)
            
            detection_boxes = detections['detection_boxes'][0].numpy()
            detection_scores = detections['detection_scores'][0].numpy()
            valid_indices = np.where(detection_scores >= threshold)[0]
            valid_boxes = detection_boxes[valid_indices]

            ground_truth_box = get_ground_truth_for_image(image_path)

            if not ground_truth_box:
                if len(valid_boxes) == 0:
                    true_negative += 1
                else:
                    false_positive += 1
                continue

            img_height, img_width, _ = image_np.shape
            normalized_gt_box = [
                ground_truth_box[1] / img_width,
                ground_truth_box[0] / img_height,
                ground_truth_box[3] / img_width,
                ground_truth_box[2] / img_height
            ]
            detected = any(compute_iou(box, normalized_gt_box) >= 0.50 for box in valid_boxes)

            if detected:
                true_positive += 1
            else:
                false_negative += 1
    
        tpr = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0
        if (false_positive + true_negative) == 0:
            fpr = 0.0
        else:
            fpr = false_positive / (false_positive + true_negative)
    
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Plot the ROC curve for each model version
    plt.figure(figsize=(10, 7))
    plt.plot(fpr_list, tpr_list, '-o', label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--')

    for i, threshold in enumerate(thresholds):
        plt.annotate(f"{threshold:.2f}", (fpr_list[i], tpr_list[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_version}')
    plt.legend()
    plt.grid(True)
    save_path = f'Calf_Detection/new_app/New_Models/graphs/extended/roc_curve_{model_version}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the plot to the specified path
    plt.savefig(save_path, bbox_inches='tight')