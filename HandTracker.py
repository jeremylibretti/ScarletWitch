import csv
import copy
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

class HandTracker:
    def __init__(self, use_static_image_mode, min_detection_confidence, min_tracking_confidence, debug=False):
        self._cap = cv.VideoCapture(0) # initialize video capture
        self.mp_hands = mp.solutions.hands

        self.running = False
        self.debug   = debug

        self.static_gesture = 0
        self.dynamic_gesture = 0
        self.hand_pos = (0, 0)

        self.use_static_image_mode = use_static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.position_tracking = False

    def run(self):
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        self.running = True

        # Model load #############################################################
        hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        keypoint_classifier = KeyPointClassifier()

        point_history_classifier = PointHistoryClassifier()

        hand_history = deque()

        finger_gesture_history = deque(maxlen=history_length)

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        # position_tracking = False
        hand_history = deque()

    ###              ###
    ### Main Process ###
    ###              ###
        while self.running and self._cap.isOpened():
        # Initialization
            fps = cvFpsCalc.get()

            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            #number, mode = self.select_mode(key, mode)

        ###           ###
        ### GET IMAGE ###
        ###           ###
            # Camera capture #####################################################
            success, image = self._cap.read()
            if not success:
                continue
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

        ###                           ###
        ### GET GESTURES AND POSITION ###
        ###                           ###
            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                    # Landmark calculation
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = self.pre_process_point_history(debug_image, point_history)
                    
                    # Write to the dataset file
                    # self.logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

            ##### Static Gesture Calculation
                    # Hand sign classification [0: Open, 1: Closed, 2: Pointer, 3: OK]
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])
                    
                    if not self.position_tracking and hand_sign_id == 1: # Closed hand
                        self.position_tracking = True
                    elif self.position_tracking and hand_sign_id != 1:
                        self.position_tracking = False
                        hand_history.clear()

            ##### Dynamic Gesture Calculation
                    # Finger gesture classification
                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                        
                    # Calculates the gesture IDs in the latest detection
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    ##### Hand Position Calculation
                    brect = self.calc_bounding_rect(debug_image, hand_landmarks)

                    x = (brect[0] + brect[2])/2
                    y = (brect[1] + brect[3])/2
                    hand_center = (round(x), round(y))

                    if self.position_tracking: hand_history.append(hand_center)

            # Update static gesture code,
            #        dynamic gesture code, and 
            #        hand position
                    self.static_gesture = hand_sign_id
                    self.dynamic_gesture = most_common_fg_id
                    self.hand_pos = hand_center

            else:
                point_history.append([0, 0])

        self._cap.release()

    def get_static_gesture(self):
        return self.static_gesture

    def get_dynamic_gesture(self):
        return self.dynamic_gesture

    def get_hand_pos(self):
        return self.hand_pos

    def get_tracking(self):
        return self.position_tracking

    def terminate(self):
        self.running = False

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        if key == 104:  # h
            mode = 2
        return number, mode

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]