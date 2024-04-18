
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import glob
import matplotlib.pyplot as plt
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import io
import glob
import absl
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
from datetime import timedelta

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from object_detection.utils import config_util
from object_detection.builders import model_builder
from datetime import timedelta, datetime

# Function to load an image into numpy array
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3)
    return img.numpy()

@tf.function
def detect_fn(image, model):
    """TF function for object detection."""
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections

def detect_objects(detection_model, image_np):
    """Detects objects in an image using the trained model."""
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)
    return detections

def load_model(pipeline_file, model_dir):
    configs = config_util.get_configs_from_pipeline_file(pipeline_file)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    
    # Restore checkpoint
    checkpoint_files = list(pathlib.Path(model_dir).glob('*.index'))
    if checkpoint_files:
        checkpoint_files.sort()
        latest_checkpoint = str(checkpoint_files[-1]).replace('.index', '')
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(latest_checkpoint).expect_partial()
    else:
        raise FileNotFoundError("No checkpoint files found in model directory.")
    
    return detection_model

def plot_detection_scores(input_dir, model_dir, pipeline_file):
    TEST_IMAGE_PATHS = sorted(glob.glob(f'{input_dir}/*.jpg'))  # Sort the image paths
    model = load_model(pipeline_file, model_dir)
    
    scores = []
    image_indices = np.arange(len(TEST_IMAGE_PATHS))  # Create an index array for the images

    for image_path in TEST_IMAGE_PATHS:
        image_np = load_image_into_numpy_array(image_path)
        detections = detect_objects(model, image_np)

        detection_scores = detections['detection_scores'].numpy()[0]
        detection_classes = detections['detection_classes'].numpy()[0]
        calf_indices = np.where(detection_classes == 1)[0]
        calf_scores = detection_scores[calf_indices]
        max_score = max(calf_scores, default=0)  # default to 0 if no calf detected
        scores.append(max_score)

    # Create time labels for the x-axis
    base_time = datetime.strptime("00:00:00", "%H:%M:%S")
    time_labels = [base_time + timedelta(minutes=int(i)) for i in image_indices]
    formatted_labels = [time.strftime('%H:%M:%S') for time in time_labels]

    # Plot the scores per image
    plt.figure(figsize=(12, 8))
    plt.bar(image_indices, scores)
    plt.xlabel('Time (hh:mm:ss)')
    plt.ylabel('Highest Detection Confidence')
    plt.title('Detection Confidence for Class "Calf" on Each Image')
    plt.xticks(image_indices[::120], formatted_labels[::120], rotation=45)  # Set x-ticks every 120 minutes
    plt.grid(True)
    plt.tight_layout()
    input_dir="/home/woody/iwso/iwso122h/Calf_Detection/new_img_data/graphs"
    img_save_path=os.path.join(input_dir,"d0_detection_scores.png")    
    plt.savefig(img_save_path)
    plt.show()

# Usage example (adjust with your actual paths)
input_img_dir = "/home/woody/iwso/iwso122h/Calf_Detection/new_img_data/img_data/2023-10-09"
model_checkpoint = "/home/woody/iwso/iwso122h/Calf_Detection/new_app/New_Models/training/epochs_100_new/efficientdet_d0_coco17_tpu-32"
pipeline_config = "/home/woody/iwso/iwso122h/Calf_Detection/new_app/New_Models/config/efficientdet_d0_coco17_tpu-32.config"
plot_detection_scores(input_img_dir, model_checkpoint, pipeline_config)
