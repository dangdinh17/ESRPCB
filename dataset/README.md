### Super-resolution Dataset
If you want to retrain the ESRPCB model, please download the pcb super-resolution dataset from [Google Drive](https://drive.google.com/file/d/10EWL9DhGOONFnoOL8ZYB19iRbCH0Vlgy/view?usp=sharing).

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
### PCB Defect Dataset
If you want to retrain defect detection model, please download the pcb dataset from [Kaggle](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset).

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