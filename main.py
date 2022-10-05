from typing import List, Tuple, cast
import cv2
import torch
from time import time, sleep
import numpy as np


feeds: List[str] = [
    "http://107.0.231.40:8082/mjpg/video.mjpg?timestamp=1664917495951",
    "http://177.72.3.203:8001/mjpg/video.mjpg?timestamp=1664917488716",
    "http://211.132.61.124/mjpg/video.mjpg",
    "http://166.143.28.201:8081/-wvhttp-01-/GetOneShot?image_size=640x480&frame_count=1000000000",
    "http://136.25.107.85:8001/mjpg/video.mjpg"
]

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    __CLASSES_TO_TRIGGER_ALARM: List[str] = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'snowboard', 'sports ball', 'skateboard',
        'bottle', 'wine glass', 'cup', 'chair', 'tv', 'laptop', 'cell phone', 'refrigerator', 'book',
    ]

    def __init__(self, url, out_file="Labeled_Video.avi"):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """

        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cpu'
        self.model.to(self.device)
        self.alarm_area=[(0, 0), (0, 0)]

    def get_video_from_url(self) -> cv2.VideoCapture:
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        return cv2.VideoCapture(self._URL)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
       #model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        return model

    def score_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """

        #self.model.to(self.device) -- MOVED TO INIT

        results = self.model([frame])

        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def isRectangleOverlap(self, r1: List[int], r2: List[int], predicted_class_name: str) -> bool:
        if predicted_class_name not in self.__CLASSES_TO_TRIGGER_ALARM:
            return False

        if (r1[0] >= r2[2]) or (r1[2] <= r2[0]) or (r1[3] <= r2[1]) or (r1[1] >= r2[3]):
            return False
        else:
            return True

    def plot_boxes(self, results: Tuple[np.ndarray, np.ndarray], frame: np.ndarray):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        invasion_detected = None

        # Process all Detections
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:

                current_class_invasion: bool = False

                # Get Prediction Name
                predicted_class_name = self.class_to_label(labels[i])

                # Check Coordinates
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                # Detect Perimeter Invasion
                if (self.isRectangleOverlap(
                        r1=[self.alarm_area[0][0], self.alarm_area[0][1], self.alarm_area[1][0], self.alarm_area[1][1]],
                        r2=[x1, y1, x2, y2],
                        predicted_class_name=predicted_class_name)
                    ):
                    invasion_detected = [(x1, y1), (x2, y2), predicted_class_name]
                    current_class_invasion = True

                # Set Detection Color
                bgr = (0, 255, 0) if not current_class_invasion else (0, 0, 255)
                border_thickness = 1 if not current_class_invasion else 2
                draw_text = f'{predicted_class_name}:{row[4]:.2f}' if not current_class_invasion else f'{predicted_class_name}:{row[4]:.2f} - INVASION'

                # Draw Detection Rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, border_thickness)
                cv2.putText(frame, draw_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, border_thickness)


        # Draw Invasion Borders
        if invasion_detected:
            cv2.putText(frame, f'Alert Zone', self.alarm_area[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, self.alarm_area[0], self.alarm_area[1], (0, 0, 255), 2)

        else:
            cv2.rectangle(frame, self.alarm_area[0], self.alarm_area[1], (255, 0, 0), 1)
            cv2.putText(frame, f'Alert Zone', self.alarm_area[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

        return frame

    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.alarm_area[0] = (x, y)

        if event == cv2.EVENT_LBUTTONUP:
            self.alarm_area[1] = (x, y)

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player: cv2.VideoCapture = self.get_video_from_url()
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.click_and_crop)

        if not player.isOpened():
            print("Player Offline")
            return

        # Get Image Dimensions and Compute the Resize Rate to 1024x768 for Optimal Model Computation
        width: int = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resize_rate: float = 1024/width

        #four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        # out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))

        while True:
            start_time = time()
            ret, frame = cast(Tuple[bool, np.ndarray], player.read())

            # Resize Frame to 1024
            new_width = int(width * resize_rate)
            new_height = int(height * resize_rate)
            frame = cv2.resize(frame, dsize=(new_width, new_height), interpolation=cv2.INTER_AREA)

            if not ret:  # Not Data
                sleep(0.1)
                continue

            results_labels = self.score_frame(frame)
            frame = self.plot_boxes(results_labels, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")

            #out.write(frame)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            sleep(0.3)


# Create a new object and execute.
a = ObjectDetection(feeds[4]) # 4
a()