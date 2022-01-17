from scarletwitch import ScarletWitch

import sys
import math
import time, os

import mediapipe as mp
import cv2 as cv
import numpy as np

import webcam
from utils import CvFpsCalc
from handtracker import HandTracker
from gestureclassifier import GestureClassifier
from collections import deque

class DynamicDataCollection(ScarletWitch):
    def __init__(self, arguments, stdout=sys.stdout):
        super().__init__(arguments, stdout)
        self.show_info = True

        self.training_mode = 2
        self.label_id = -1
        self.record = False
        self.last_key = ""
        self.actions = ["none", "click", "double click"]
        self.datasets = [[] for i in range(len(self.actions))]

        self.secs_for_action = 2
        self.min_freq = self.secs_for_action*30
        self.last_label = -1

        self.time_created = int(time.time())
        self.time_0 = time.time()
        self.start_time = time.time()

        self.draw_error = False

        os.makedirs('dataset', exist_ok=True)

    def key_action(self, key):
        self.draw_error = False

        if self.last_label < 0:
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

            if (self.label_id+1) > len(self.actions):
                self.draw_error = True
                self.label_id = -1

            if not self.record:
                self.last_label = self.label_id

        if key == 114 and not self.record: # r (record)
            self.record = True
            self.start_time = time.time()
            if self.last_label >= 0:
                self.datasets[self.label_id].append([])
            else:
                self.draw_error = True

        if key == 110: # n (none)
            self.label_id = -1
            if not self.record:
                self.last_label = self.label_id

        if self.record:
            times_up = (time.time() - self.start_time) > self.secs_for_action
            if times_up:
                self.record = False

    def process_landmarks(self, debug_img, landmarks, history, classifier):
        # Bounding box calculation
        self.brect = self.calc_bounding_rect(debug_img, landmarks)

        # Joint angle calculations
        landmark_data = self.landmarks_to_angles(landmarks)
        history.append(landmark_data)

        if self.last_label >= 0 and self.record:
            # print("added to dataset")
            self.datasets[self.label_id][-1].append(landmark_data)
    
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
        start_time = self.start_time

        # FPS
        cv.putText(image, "FPS:" + str(fps), (10, 35), cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Training
        training_text = ["Off", "Static", "Dynamic"]
        cv.putText(image, "Training: " + training_text[self.training_mode], (10, 70),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Label id
        id = str(self.last_label)
        text_color = (255, 255, 255)
        sum = 0

        if self.last_label == -1: # use last_label instead of label_id
            id = "None"
        else:
            sum = len(self.datasets[self.last_label])

        if self.draw_error:
            text_color = (0, 0, 255)
            
        cv.putText(image, "Label ID: " + id, (10, 110),
                cv.FONT_HERSHEY_DUPLEX, 1.0, text_color, 1, cv.LINE_AA)

        # Number of gestures recorded
        cv.putText(image, "Sum: " + str(sum) + "", (10, 145),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Key Press
        cv.putText(image, self.last_key, (dimensions[0] - 35, 35),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Recording dynamic gesture
        if self.record and self.last_label >= 0:
                cv.putText(image, "RECORDING: ", (10, 180),
                        cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)
                time_left = self.secs_for_action - (time.time() - start_time)
                cv.putText(image, str(time_left), (15, 205), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        return image

    def terminate(self):
        time = self.time_created
        # Write to the dataset files
        
        # Find length of shortest string of data
        for d in range(len(self.datasets)):
            data = self.datasets[d]
            if len(data) > 0:
                for seq in range(len(data)):
                    seq_data = data[seq]
                    if len(seq_data) < self.min_freq:
                        self.min_freq = len(seq_data)

        self.min_freq = math.floor(self.min_freq*0.95)

        # Write files
        for d in range(len(self.datasets)):
            data = self.datasets[d]
            if len(data) > 0:
                data = np.array(data)
                # print(action, data.shape)
                np.save(os.path.join('dataset', 'test_data_', f'raw_{self.actions[d]}_{time}'), data)

                # Create sequence data
                full_seq_data = []
                for seq in range(len(data)):
                    seq_data = data[seq]
                    seq_data = seq_data[0: self.min_freq]
                    full_seq_data.append(seq_data)
                
                full_seq_data = np.array(full_seq_data)
                # print(action, full_seq_data.shape)
                np.save(os.path.join('dataset', 'test_data_', f'seq_{self.actions[d]}_{time}'), full_seq_data)

                print("wrote dynamic data")

        print("min_freq = " + str(self.min_freq))

        self.running = False