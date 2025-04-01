# ESRPCB
### Introduction
This is an official implementation for "ESRPCB: an Edge guided Super - Resolution model and Ensemble learning for tiny Printed Circuit Board Defect detection".

## 1. Create Environment
### 1.1. Install the environment
Make Conda Environment
```
conda env create -f setup.txt
conda activate pcb
```
Note: If you want to rename the new environment, edit the [`setup.txt`](setup.txt) file by changing the first line:
```
name: <new environment name>
```

## 2. Dataset
### 2.1 Super-resolution Dataset
If you want to retrain the ESRPCB model, please download the pcb super-resolution dataset from [Google Drive](https://drive.google.com/file/d/10EWL9DhGOONFnoOL8ZYB19iRbCH0Vlgy/view?usp=sharing)

The structure of the dataset will be like this:
```
dataset/
    sr-dataset/
        train/
            HR/
            LR/
        test/
            HR/
            LR/

```
### 2.2 PCB Defect Dataset
If you want to retrain defect detection model, please download the pcb dataset from [Kaggle](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset)

The structure of the dataset will be like this:
```
dataset/
    pcbdefect/
        train/
            images/
            labels/
        val/
            images/
            labels/
        test/
            images/
            labels/

```

## 3. Training
To train the ESRPCB model, use the file [train_esrpcb.ipynb](train_esrpcb.ipynb).

To train the YOLOv8 and YOLOv9 model, use the file [train_yolo.ipynb](train_yolo.ipynb).
