import torch
import torch
from ensemble_boxes import *
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yaml
import os
import matplotlib.pyplot as plt
import random 
import cv2
import ultralytics
from ultralytics import YOLO
from tqdm import tqdm
from models.utils import *
from models.detection import *

yolov8_path = 'outputs/weight_detection/best_yolov8.pt'
yolov9_path = 'outputs/weight_detection/best_yolov8.pt'
yaml_path = 'data.yaml'
ensemble = Ensemble(weight_yolov8=yolov8_path, weight_yolov9=yolov9_path, ensemble_type='WBF', yaml_path=yaml_path)
ensemble.predict('dataset/test1_600x600\images\l_light_01_missing_hole_04_2_600.jpg')