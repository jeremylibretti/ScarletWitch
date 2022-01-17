from scarletwitch import ScarletWitch

import sys
import time, os

import mediapipe as mp
import cv2 as cv
import numpy as np

import webcam
from utils import CvFpsCalc
from handtracker import HandTracker
from gestureclassifier import GestureClassifier
from collections import deque

class StaticDataCollection(ScarletWitch):
    def __init__(self, arguments, stdout=sys.stdout):
        super().__init__(arguments, stdout)
        self.show_info = True

        self.training_mode = 1
        self.label_id = -1
        self.record = False
        self.last_key = ""
        self.actions = ['open', 'closed', 'pointer', 'ok', 'peace', 'thumbs up']
        self.datasets = [[] for i in range(10)]
        self.time_created = int(time.time())

        os.makedirs('dataset', exist_ok=True)

    def key_action(self, key):
        self.record = False

        if key > -1:
            self.last_key = chr(key)
        else:
            self.last_key = ""

        if key == 27:  # ESC
            self.last_key = "ESC"
            self.time_created = int(time.time())
            self.terminate()

        if 48 <= key <= 57:  # 0 ~ 9
            self.label_id = key - 48

        if key == 114: # r (record)
            self.record = True

        if key == 110: # n (none)
            self.label_id = -1

    def process_landmarks(self, debug_img, landmarks, history, classifier):
        # Bounding box calculation
        self.brect = self.calc_bounding_rect(debug_img, landmarks)

        # Joint angle calculations
        landmark_data = self.landmarks_to_angles(landmarks)
        history.append(landmark_data)

        if self.label_id >= 0 and self.record:
            self.datasets[self.label_id].append(landmark_data)
    
        static_gesture_id, static_gesture_label = classifier.classify_static(landmark_data)
        dynamic_gesture_id, dynamic_gesture_label = classifier.classify_dynamic(history)

        # Image Drawing
        debug_image = self.draw_bounding_rect(debug_img, self.brect)

        self.mp_drawing.draw_landmarks(debug_image, landmarks, self.mp_hands.HAND_CONNECTIONS)
        debug_image = self.draw_hand_info(debug_image, self.brect, static_gesture_label)

        return history, [static_gesture_id, static_gesture_label], [dynamic_gesture_id, dynamic_gesture_label]

    def draw_hand_info(self, image, brect, static_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)

        info_text = static_gesture_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        return image
    
    def draw_window_info(self, image, dimensions, fps, dynamic_gesture_text):
        # FPS
        cv.putText(image, "FPS:" + str(fps), (10, 35), cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Training
        training_text = ["Off", "Static", "Dynamic"]
        cv.putText(image, "Training: " + training_text[self.training_mode], (10, 70),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Label id
        id = str(self.label_id)
        text_color = (255, 255, 255)

        if self.label_id == -1:
            id = "None"
            if self.record:
                text_color = (0, 0, 255)
        
        cv.putText(image, "Label ID: " + id, (10, 110),
                cv.FONT_HERSHEY_DUPLEX, 1.0, text_color, 1, cv.LINE_AA)

        # Number of times recorded
        cv.putText(image, "Sum: " + str(len(self.datasets[self.label_id])) + "", (10, 145),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Key Press
        cv.putText(image, self.last_key, (dimensions[0] - 35, 35),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        return image

    def terminate(self):
        time = self.time_created

        # Write to the dataset files
        for d in range(len(self.datasets)):
            data = self.datasets[d]
            if len(data) > 0:
                data = np.array(data)
                np.save(os.path.join('dataset', 'test_data_', f'raw_{self.actions[d]}_{time}'), data)
                print("Wrote static data")

        self.running = False