import numpy as np
import json
import os
import cv2
import time
import yaml
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from modules.sort import Sort

class YOLOv8_ObjectDetector:

    def __init__(self, model_file='yolov8n.pt', labels=None, classes=None, conf=0.25, iou=0.45):
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.model = YOLO(model_file)
        self.model_name = model_file.split('.')[0]
        self.results = None
        self.labels = labels if labels is not None else self.model.names

    def predict_img(self, img, verbose=True):
        results = self.model(img, classes=self.classes, conf=self.conf, iou=self.iou, verbose=verbose)
        self.orig_img = img
        self.results = results[0]
        return results[0]

    def default_display(self, show_conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        if self.results is None:
            raise ValueError('No detected objects to display. Call predict_img() method first.')
        display_img = self.results.plot(show_conf, line_width, font_size, font, pil, example)
        return display_img

    def custom_display(self, colors, show_cls=True, show_conf=True):
        img = self.orig_img
        bbx_thickness = (img.shape[0] + img.shape[1]) // 450
        for box in self.results.boxes:
            textString = ""
            score = box.conf.item() * 100
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)
            if show_cls:
                textString += f"{self.labels[class_id]}"
            if show_conf:
                textString += f" {score:,.2f}%"
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = (((x2 - x1) / img.shape[0]) + ((y2 - y1) / img.shape[1])) / 2 * 2.5
            fontThickness = 1
            textSize, baseline = cv2.getTextSize(textString, font, fontScale, fontThickness)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_id], bbx_thickness)
            center_coordinates = ((x1 + x2) // 2, (y1 + y2) // 2)
            img = cv2.circle(img, center_coordinates, 5, (0, 0, 255), -1)
            if textString != "":
                if (y1 < textSize[1]):
                    y1 = y1 + textSize[1]
                else:
                    y1 -= 2
                img = cv2.rectangle(img, (x1, y1), (x1 + textSize[0], y1 - textSize[1]), colors[class_id], cv2.FILLED)
                img = cv2.putText(img, textString, (x1, y1), font, fontScale, (0, 0, 0), fontThickness, cv2.LINE_AA)
        return img

    def predict_video(self, video_path, save_dir, save_format="avi", display='custom', verbose=True, **display_args):
        cap = cv2.VideoCapture(video_path)
        vid_name = os.path.basename(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir+f"/{vid_name.split('.')[0]}"):
            os.makedirs(save_dir+f"/{vid_name.split('.')[0]}")

        save_name = self.model_name + ' -- ' + vid_name.split('.')[0] + '.' + save_format
        save_file = os.path.join(save_dir, save_name)

        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vid_name} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + save_file)

        out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))

        if not cap.isOpened():
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            beg = time.time()
            results = self.predict_img(frame, verbose=False)
            if results is None:
                print('***********************************************')
            fps = 1 / ((time.time() - beg) + 0.000001)

            if display == 'default':
                frame = self.default_display(**display_args)
            elif display == 'custom':
                frame = self.custom_display(**display_args)

            frame = cv2.putText(frame, f"FPS : {fps:,.2f}", (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()

class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):

    def __init__(self, model_file='yolov8n.pt', labels=None, classes=None, conf=0.25, iou=0.45,
                 track_max_age=45, track_min_hits=15, track_iou_threshold=0.3, trail_length=30):

        super().__init__(model_file, labels, classes, conf, iou)
        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold
        self.trail_length = trail_length  # Length of the trail
        self.positions_history = {}  # Dictionary to store position history for each object

    
    def predict_video(self, video_path, save_dir, save_format="avi", display='custom', verbose=True, **display_args):
        cap = cv2.VideoCapture(video_path)
        vid_name = os.path.basename(video_path)
        width = int(cap.get(3))
        height = int(cap.get(4))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir+f"/{vid_name.split('.')[0]}"):
            os.makedirs(save_dir+f"/{vid_name.split('.')[0]}")

        save_name = vid_name.split('.')[0] + '.' + save_format
        save_file = os.path.join(save_dir, save_name)
        yaml_save_file = os.path.join(save_dir, vid_name.split('.')[0] + '.yaml')

        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vid_name} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + save_file)
            print('SAVING ANNOTATIONS TO :' + yaml_save_file)

        out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))

        if not cap.isOpened():
            print("Error opening video stream or file")

        tracker = Sort(max_age=self.track_max_age, min_hits=self.track_min_hits, iou_threshold=self.track_iou_threshold)
        totalCount = []
        currentArray = np.empty((0, 5))
        annotations = []  # List to store annotations for each frame
        object_confidences = {}  # Dictionary to track the highest confidence for each object

        while cap.isOpened():
            detections = np.empty((0, 5))
            ret, frame = cap.read()

            if not ret:
                break

            beg = time.time()
            results = self.predict_img(frame, verbose=False)
            if results is None:
                print('***********************************************')
            fps = 1 / ((time.time() - beg) + 0.00000001)
            
            frame_annotations = {'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 'objects': []}

            for box in results.boxes:
                score = box.conf.item() * 100
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)
                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))

            resultsTracker = tracker.update(detections)
            new_positions = {}  # Dictionary to store new positions for this frame

            for result in resultsTracker:
                x1, y1, x2, y2, obj_id = result
                x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)

                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                # Update position history
                if obj_id not in self.positions_history:
                    self.positions_history[obj_id] = []
                self.positions_history[obj_id].append((cx, cy))

                # Limit the history length
                if len(self.positions_history[obj_id]) > self.trail_length:
                    self.positions_history[obj_id].pop(0)

                # Track highest confidence score for each object
                if obj_id not in object_confidences:
                    object_confidences[obj_id] = {'score': score, 'bbox': (x1, y1, x2, y2), 'frame': frame.copy()}
                elif score > object_confidences[obj_id]['score']:
                    object_confidences[obj_id] = {'score': score, 'bbox': (x1, y1, x2, y2), 'frame': frame.copy()}

                # Draw the movement trail
                if len(self.positions_history[obj_id]) > 1:
                    for i in range(len(self.positions_history[obj_id]) - 1):
                        pt1 = self.positions_history[obj_id][i]
                        pt2 = self.positions_history[obj_id][i + 1]
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green trail line

                id_txt = f"ID: {str(obj_id)}"
                cv2.putText(frame, id_txt, (cx, cy), 4, 0.5, (0, 0, 255), 1)

                if obj_id not in totalCount:
                    totalCount.append(obj_id)

                # Collect annotations
                frame_annotations['objects'].append({
                    'id': obj_id,
                    'bbox': [x1, y1, x2, y2],
                    'score': score
                })

            annotations.append(frame_annotations)

            if display == 'default':
                frame = self.default_display(**display_args)
            elif display == 'custom':
                frame = self.custom_display(**display_args)

            frame = cv2.putText(frame, f"FPS : {fps:,.2f}", (5, 55), cv2.FONT_HERSHEY_COMPLEX, 
                                0.5, (0, 255, 255), 1, cv2.LINE_AA)
            count_txt = f"TOTAL COUNT : {len(totalCount)}"
            frame = cv2.putText(frame, count_txt, (5, 45), cv2.FONT_HERSHEY_COMPLEX, 2, 
                                (0, 0, 255), 2)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()

        # Save one frame per unique object based on the highest confidence level
        saved_objects_count = 0
        for obj_id, data in object_confidences.items():
            x1, y1, x2, y2 = data['bbox']
            
            # Ensure bounding box coordinates are within image dimensions
            x1 = max(0, min(x1, data['frame'].shape[1] - 1))
            y1 = max(0, min(y1, data['frame'].shape[0] - 1))
            x2 = max(x1 + 1, min(x2, data['frame'].shape[1]))
            y2 = max(y1 + 1, min(y2, data['frame'].shape[0]))
            
            # Crop the object from the frame
            cropped_object = data['frame'][y1:y2, x1:x2]
            
            # Check if the cropped image is not empty
            if cropped_object.size == 0:
                print(f"Skipped saving object {obj_id} due to empty crop.")
                continue
            
            object_save_path = os.path.join(save_dir+f"/{vid_name.split('.')[0]}", f"object_{obj_id}.jpg")
            
            # Save the cropped image
            success = cv2.imwrite(object_save_path, cropped_object)
            if success:
                saved_objects_count += 1
            else:
                print(f"Failed to save object {obj_id} at {object_save_path}")

        yaml_data = {
            'video': vid_name,
            'parameters': {
                'model_file': self.model_name,
                'conf_threshold': self.conf,
                'iou_threshold': self.iou,
                'track_max_age': self.track_max_age,
                'track_min_hits': self.track_min_hits,
                'track_iou_threshold': self.track_iou_threshold,
                'trail_length': self.trail_length
            },
            'total_count': len(totalCount),
            'counts': totalCount,
            'annotations': annotations,
            'saved_objects_count': saved_objects_count
        }

        with open(yaml_save_file, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        print(f"Total saved objects: {saved_objects_count}")
        print(len(totalCount))