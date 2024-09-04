import random
from modules.counter import YOLOv8_ObjectCounter

def main():
    yolo_names = ['model.pt', 'yolov8l.pt', 'yolov8m.pt', 'yolov8n.pt', 'yolov8s.pt']
    
    colors = []
    for _ in range(80):
        rand_tuple = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        colors.append(rand_tuple)

    counters = []
    for yolo_name in yolo_names:
        counter = YOLOv8_ObjectCounter(yolo_name, conf = 0.60 )
        counters.append(counter)

    for counter in counters:
        counter.predict_video(video_path= '2_HD.mp4', save_dir = 'test', save_format = "mp4", display = 'custom', colors = colors)

if __name__=='__main__':
    main()