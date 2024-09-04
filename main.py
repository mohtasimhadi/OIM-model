import random
import os
import argparse
from modules.counter import YOLOv8_ObjectCounter

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Counter")
    parser.add_argument('--model', type=str, default='model.pt', help="YOLO model file name (default: model.pt)")
    parser.add_argument('--video', type=str, default='videos', help="Path to the video file or folder containing videos")
    parser.add_argument('--conf', type=float, default=0.60, help="Confidence threshold (default: 0.60)")
    parser.add_argument('--save_dir', type=str, default='results', help="Directory to save the results (default: results)")
    parser.add_argument('--save_format', type=str, default='mp4', help="Format to save the output video (default: mp4)")
    return parser.parse_args()

def generate_colors(num_colors=80):
    colors = []
    for _ in range(num_colors):
        rand_tuple = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        colors.append(rand_tuple)
    return colors

def process_video(counter, video_path, save_dir, save_format, colors):
    if os.path.isfile(video_path):
        counter.predict_video(video_path=video_path, save_dir=save_dir, save_format=save_format, display='custom', colors=colors)
    elif os.path.isdir(video_path):
        for video_file in os.listdir(video_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                full_path = os.path.join(video_path, video_file)
                counter.predict_video(video_path=full_path, save_dir=save_dir, save_format=save_format, display='custom', colors=colors)
    else:
        print(f"[Error] Invalid video path: {video_path}")

def main():
    args = parse_arguments()

    colors = generate_colors()
    counter = YOLOv8_ObjectCounter(f'models/{args.model}', conf=args.conf)

    process_video(counter, args.video, args.save_dir, args.save_format, colors)

if __name__ == '__main__':
    main()