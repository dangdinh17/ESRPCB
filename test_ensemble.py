import torch
import torch
import numpy as np
from models.utils import *
from models.detection import *

yolov8_path = 'outputs/weight_detection/best_yolov8.pt'
yolov9_path = 'outputs/weight_detection/best_yolov8.pt'
yaml_path = 'data.yaml'
ensemble = Ensemble(weight_yolov8=yolov8_path, weight_yolov9=yolov9_path, ensemble_type='WBF', yaml_path=yaml_path)
ensemble.predict('dataset/l_light_01_missing_hole_04_2_600.jpg')