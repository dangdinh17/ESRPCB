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
from models.utils import *

class Ensemble():
    def __init__(self, weight_yolov8=None, weight_yolov9=None, ensemble_type='WBF', yaml_path = './data.yaml'):
        
        self.yolov8 = None
        self.yolov9 = None
        self.method = ensemble_type
        self.output = './run'
        self.val_path = None
        os.makedirs(self.output, exist_ok=True)
        try:
            self.yolov8 = YOLO(weight_yolov8)
            # print("YOLOv8 model loaded successfully.")
        except Exception as e:
            self.yolov8 = YOLO('./outputs/weight_detection/best_yolov8.pt')
            print(f"Failed to load YOLOv8: {e}")

        try:
            self.yolov9 = YOLO(weight_yolov9)
            # print("YOLOv9 model loaded successfully.")
        except Exception as e:
            self.yolov9 = YOLO('./outputs/weight_detection/best_yolov9.pt')
            print(f"Failed to load YOLOv9: {e}")
            
        try:
            with open(yaml_path, "r") as file:
                data = yaml.safe_load(file)
            self.class_names = data['names']
            self.label_colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(data['nc'])}
            self.val_path = data['val']
        except Exception as e:
            self.class_names = {
                0: 'mouse_bite',
                1: 'spur',
                2: 'missing_hole',
                3: 'short',
                4: 'open_circuit',
                5: 'spurious_copper'
            }

            self.label_colors = {
                0: (255, 0, 0),  # Red
                1: (0, 255, 0),  # Green
                2: (0, 0, 255),  # Blue
                3: (255, 255, 0),  # Yellow
                4: (255, 0, 255),  # Magenta
                5: (0, 255, 255)   # Cyan
            }
        

    def ensemble_inference(self, image, method):
        # Get predictions from both models
        boxes_yolov8, scores_yolov8, labels_yolov8 = run_inference(image, self.yolov8)
        boxes_yolov9, scores_yolov9, labels_yolov9 = run_inference(image, self.yolov9)
        # print(boxes_yolov8)
        # Get image size
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:  # PIL Image
            width, height = image.size
        image_size = (width, height)
        if method in ['YOLOv8', 'YOLOv9']:
            if method == 'YOLOv8':
                boxes, scores, labels = boxes_yolov8, scores_yolov8, labels_yolov8
            else:
                boxes, scores, labels = boxes_yolov9, scores_yolov9, labels_yolov9
        else:
            boxes_yolov8_norm = normalize_boxes(boxes_yolov8, image_size)
            boxes_yolov9_norm = normalize_boxes(boxes_yolov9, image_size)
            boxes_list = [boxes_yolov8_norm, boxes_yolov9_norm]
            scores_list = [scores_yolov8, scores_yolov9]
            labels_list = [labels_yolov8, labels_yolov9]
            if len(boxes_list) == 0 or all(len(b) == 0 for b in boxes_list):
                # print("Warning: No valid boxes detected. Returning empty results.")
                boxes = np.empty((0, 4), dtype=np.float32)  # Dạng bounding box
                scores = np.empty((0,), dtype=np.float32)  # Dạng scores
                labels = np.empty((0,), dtype=np.int32)
            else:
                # print(boxes_list, scores_list)
                if method == 'WBF':
                    # Apply Weighted Box Fusion
                    boxes, scores, labels = weighted_boxes_fusion(
                        boxes_list,
                        scores_list,
                        labels_list,
                        weights=[1, 2],  # Equal weights for both models
                        iou_thr=0.5,
                        skip_box_thr=0.0001,
                    )
                elif method == 'NMS':
                    boxes, scores, labels = nms(
                        boxes_list,
                        scores_list,
                        labels_list,
                        weights=[1, 2],  # Equal weights for both models
                        iou_thr=0.5
                    )
                    # print(type(boxes), '\n', type(scores), '\n',type(labels))
                elif method == 'Soft-NMS':
                    boxes, scores, labels = soft_nms(
                        boxes_list,
                        scores_list,
                        labels_list,
                        weights=[1, 2],  # Equal weights for both models
                        iou_thr=0.5,
                        sigma = 0.1,
                        thresh = 0.0001
                    )
                else:
                    boxes, scores, labels = non_maximum_weighted(
                        boxes_list,
                        scores_list,
                        labels_list,
                        weights=[1, 2],  # Equal weights for both models
                        iou_thr=0.5,
                        skip_box_thr=0.0001,
                    )
                boxes = denormalize_boxes(boxes, image_size)
        # Convert back to torch tensors
        boxes = torch.from_numpy(boxes).float()
        scores = torch.from_numpy(scores).float()
        labels = torch.from_numpy(labels).long()
        return boxes, scores, labels

    def read_label_file(self, label_path, img_width, img_height):
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                box = list(map(float, line.strip().split()))
                labels.append(int(box[0]))
                boxes.append(yolo_to_xyxy(box, img_width, img_height))
        return boxes, labels

    def predict(self, img_path=None):
        image = Image.open(img_path).convert('RGB')
        img_name = os.path.basename(img_path)
  
        pred_boxes, pred_scores, pred_labels = self.ensemble_inference(image, self.method)
       
        os.makedirs(os.path.join(self.output + '/predict'), exist_ok=True)
        save_path = os.path.join(self.output + '/predict', img_name)  # Lưu với cùng tên ảnh
        draw_and_save_predictions(
            image=image.copy(),
            boxes=pred_boxes.numpy(),
            labels=pred_labels.numpy(),
            scores=pred_scores.numpy(),
            class_names=self.class_names,
            label_colors=self.label_colors,
            save_path=save_path
        )
        print(f'Predicted image save at {save_path}.')
    
    def val(self):
        map_metric = MeanAveragePrecision(iou_thresholds=[0.5], iou_type="bbox", class_metrics=True)
        
        method_output_dir = os.path.join(self.output + '/val')
        os.makedirs(method_output_dir, exist_ok=True)
        print('Start validation......')
        for image_file in tqdm(os.listdir(self.val_path), unit='img'):
            img_path = os.path.join(self.val_path, image_file)
            image = Image.open(img_path).convert('RGB')

            # Get true labels
            labels_dir = os.path.dirname(self.val_path) + '/labels'
            label_path = os.path.join(labels_dir, image_file[:-4] + ".txt")
            true_boxes, true_labels = self.read_label_file(label_path, image.size[0], image.size[1])
            true_boxes = torch.tensor(true_boxes, dtype=torch.float32)
            true_labels = torch.tensor(true_labels, dtype=torch.int64)

            # Chạy inference theo phương pháp
            pred_boxes, pred_scores, pred_labels = self.ensemble_inference(image, self.method)

            # Format predictions and targets for metric update
            predictions = [{
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            }]
            
            targets = [{
                'boxes': true_boxes,
                'labels': true_labels
            }]

            map_metric.update(predictions, targets)

            save_path = os.path.join(method_output_dir, image_file)  # Lưu với cùng tên ảnh
            draw_and_save_predictions(
                image=image.copy(),
                boxes=pred_boxes.numpy(),
                labels=pred_labels.numpy(),
                scores=pred_scores.numpy(),
                class_names=self.class_names,
                label_colors=self.label_colors,
                save_path=save_path
            )

        # Tính toán kết quả sau khi xử lý tất cả ảnh
        results = map_metric.compute()
        map50_per_class = results['map_per_class']
        for class_idx, map_value in enumerate(map50_per_class):
            print(f"Class {class_idx}: mAP@50 = {map_value.item():.3f}")
        print(results['map'])
        print("-" * 50)