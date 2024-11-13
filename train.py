import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-C2f-Star-RDCN-LSCD.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/linzeyu/programs/experiments/ultralytics-main/data-NEU.yaml',
                cache=True,
                epochs=300,
                patience=300,
                imgsz=640,
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='v8-C2f-Star-RDCN-LSCD-NEU_3',
                )