# importing OpenCV, time and Pandas library
import cv2, time, pandas
from typing import Tuple, List
# importing datetime class from datetime library
from datetime import datetime
import numpy as np

class MotionDetector:

    def __init__(self, sensibility: int = 500) -> None:

        self.current_frame = None
        self.previous_frame = None
        self.sensibility = sensibility

    def detect_motion(self, frame) -> Tuple[bool, List]:
        has_motion = False

        self.current_frame = frame

        # Converting color image to gray_scale image
        gray_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        # Converting gray scale image to GaussianBlur
        # so that change can be find easily
        gray = cv2.GaussianBlur(src=gray_frame, ksize=(5, 5), sigmaX=0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return False, None

        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=self.previous_frame, src2=gray_frame)
        self.previous_frame = gray_frame

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        # 5. Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        h_result: List = []
        for contour in contours:
            if cv2.contourArea(contour) < self.sensibility:
                # too small: skip!
                continue

            has_motion = True
            (x, y, w, h) = cv2.boundingRect(contour)
            h_result.append((x, y, w, h))
        #
        # # Displaying image in gray_scale
        # cv2.imshow("Gray Frame", gray_frame)
        #
        # # Displaying image in gray_scale
        # cv2.imshow("Previous Frame", self.previous_frame)
        #
        # # Displaying the difference in currentframe to
        # # the staticframe(very first_frame)
        # cv2.imshow("Difference Frame", diff_frame)
        #
        # # Displaying the black and white image in which if
        # # intensity difference greater than 30 it will appear white
        # cv2.imshow("Threshold Frame", thresh_frame)
        #
        # # Displaying color frame with contour of motion of object
        # cv2.imshow("Color Frame", self.current_frame)

        return has_motion, h_result
