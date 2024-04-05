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
# Training
command = [
    'python3',
    'Calf_Detection/models/research/object_detection/model_main_tf2.py',
    '--pipeline_config_path=Calf_Detection/new_app/config/efficientdet_d0_coco17_tpu-32.config',
    '--model_dir=Calf_Detection/new_app/training_ed0_epochs_100',
    '--alsologtostderr'
]

subprocess.run(command, check=True)

pipeline_file = 'Calf_Detection/new_app/config/efficientdet_d0_coco17_tpu-32.config'
last_model_path = 'Calf_Detection/new_app/training_ed0_epochs_100'
output_directory = "Calf_Detection/new_app/finetuned_ed0_epochs_100"

# Export
command = [
    'python3',
    'Calf_Detection/models/research/object_detection/exporter_main_v2.py',
    '--trained_checkpoint_dir', last_model_path,
    '--output_directory', output_directory,
    '--pipeline_config_path', pipeline_file
]

subprocess.run(command, check=True)

# Evaluation
pipeline_file = 'Calf_Detection/new_app/config/efficientdet_d0_coco17_tpu-32.config'
model_directory = 'Calf_Detection/new_app/training_ed0_epochs_100'
checkpoint_directory = 'Calf_Detection/new_app/training_ed0_epochs_100'

command = [
    'python3',
    'Calf_Detection/models/research/object_detection/model_main_tf2.py',
    '--pipeline_config_path', pipeline_file,
    '--model_dir', model_directory,
    '--checkpoint_dir', checkpoint_directory,
    '--alsologtostderr'
]

subprocess.run(command, check=True)