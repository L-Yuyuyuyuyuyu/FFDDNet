import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/linzeyu/programs/experiments/ultralytics-main/runs/train/'
                'v8-C2f-Star-CAA_NEU_3/weights/best.pt')
    metrics = model.val(data='/home/linzeyu/programs/experiments/ultralytics-main/data-NEU.yaml',
                        split='val',
                        imgsz=640,
                        batch=16,
                        # rect=False,
                        save_json=True, # if you need to cal coco metrice
                        project='runs/val',
                        name='exp',
                        )
    result = metrics.box.maps
    result1 = metrics.box.map50
    result2 = metrics.box.map75
    print("maps:",result)
    print('map50:',result1)
    print('map75', result2)