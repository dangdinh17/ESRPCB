import torch
from torchvision.ops import nms
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

def draw_and_save_predictions(image, boxes, labels, scores, class_names, label_colors, save_path=None, font_path=None):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Tải font (nếu có)
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size=25)
        except Exception as e:
            print(f"Không thể tải font từ {font_path}. Sử dụng font mặc định.")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    # Vẽ từng bounding box
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        label_text = f"{class_names.get(label.item(), 'Unknown')} {score:.2f}"
        color = label_colors.get(label.item())
        # Vẽ hình chữ nhật
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Vẽ nhãn với nền
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        # else:
            # text_width, text_height = draw.textsize(label_text, font=font)
        draw.rectangle(
            [x1, y1 - text_height, x1 + text_width, y1],
            fill=color
        )
        draw.text((x1, y1 - text_height), label_text, fill="white", font=font)
    
    # Lưu ảnh nếu `save_path` được cung cấp
    if save_path:
        draw_image.save(save_path)

def yolo_to_xyxy(box, img_width, img_height):
    cx, cy, w, h = box[1:]
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return [x1, y1, x2, y2]

def run_inference(image, model):
    results = model.predict(image, verbose=False)
    predictions = results[0].boxes
    # Convert to numpy for WBF compatibility
    boxes = predictions.xyxy.cpu().numpy()
    scores = predictions.conf.cpu().numpy()
    labels = predictions.cls.cpu().numpy()
    return boxes, scores, labels

def normalize_boxes(boxes, image_size):
    """Normalize box coordinates to [0, 1] range"""
    width, height = image_size
    normalized_boxes = boxes.copy()
    normalized_boxes[:, [0, 2]] /= width
    normalized_boxes[:, [1, 3]] /= height
    return normalized_boxes

def denormalize_boxes(boxes, image_size):
    """Convert normalized boxes back to pixel coordinates"""
    width, height = image_size
    denormalized_boxes = boxes.copy()
    denormalized_boxes[:, [0, 2]] *= width
    denormalized_boxes[:, [1, 3]] *= height
    return denormalized_boxes