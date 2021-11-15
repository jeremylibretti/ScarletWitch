# Display camera view with annotations
# Execute commands given gestures

import sys
import copy
import csv
import itertools
import keyboard
import pyautogui
import win32gui
import win32con
import math

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from handtracker import HandTracker
from gestureclassifier import GestureClassifier
from webcam import WebcamVideoStream
from window import WindowMgr
from collections import deque

class ScarletWitch:
    def __init__(self, arguments, stdout=sys.stdout):
        self.running = False
        self.args = arguments
        self.stdout = stdout

        self.brect = [0, 0, 0, 0]
        self.hand_pos_tracking = False
        self.mode_view = True
        self.prev = (0, 0)
        self.curr = (0, 0)
        self.last_key = ""
        self.show_information = False

        keyboard.add_hotkey('.', self.terminate)

        self.w = WindowMgr()

    def run(self):
        self.running = True

        empty_landmarks = [[0, 0] for x in range(21)]

    # Argument deconstruction
        cap_device = self.args.device
        cap_width = self.args.width
        cap_height = self.args.height

        self.training = self.args.collect_static or self.args.collect_dynamic

        use_brect = True

    # Camera preparation
        vs = WebcamVideoStream(cap_width, cap_height, cap_device).start()

    # FPS Measurement Initialization
        cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Set stdout
        stdout = sys.stdout
        sys.stdout = self.stdout

    # Manage Windows
        self.w.find_window_wildcard(".*Minecraft 1.9.*")
        if self.w.get_handle() != None:
            self.w.set_foreground()
            self.w.maximize()

        winname = "Scarlet Witch"
        if self.training:
            winname += " Data Collection"

        dims = [[640, 360], [1280, 720]]
        win_dims = dims[0]

        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.resizeWindow(winname, win_dims[0], win_dims[1])
        cv.moveWindow(winname, 0,0)

        self.w.find_window_wildcard(".*Scarlet Witch.*")
        if self.w.get_handle() != None:
            win32gui.SetWindowPos(self.w.get_handle(), win32con.HWND_TOPMOST, 0,0,0,0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    # Coordinate history
        dynamic_data_length = math.ceil((self.get_data_length()-1)/42)
        history_length = max(dynamic_data_length, 128)
        if self.training:
            history_length = None
        landmark_history = deque(maxlen=history_length)

    # Create HandTracker, GestureClassifier
        tracker = HandTracker(self.args, vs).start()
        classifier = GestureClassifier(history_length)
        dynamic_gesture_label = ""

    # MediaPipe hands model
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

    # Main Loop
        while self.running:
        # Get fps
            fps = cvFpsCalc.get()

        # Process Key
            key = cv.waitKey(10)

            if key > -1:
                self.last_key = chr(key)
            else:
                self.last_key = ""

            if key == 27:  # ESC
                self.last_key = "ESC"
                self.terminate()

            if key == 105: # i
                self.show_information = not self.show_information
                win_dims = dims[self.show_information]
                cv.resizeWindow(winname, win_dims[0], win_dims[1])


        # Camera capture
            image = vs.read()
            tracking_results = tracker.read()

            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

        # Landmark processing
            if tracking_results.multi_hand_landmarks is not None:
                for hand_landmarks in tracking_results.multi_hand_landmarks:
                # Bounding box calculation
                    self.brect = self.calc_bounding_rect(debug_image, hand_landmarks)

                # Landmark calculation
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(hand_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])

        # Gesture classification
                # Static
                    static_gesture_id, static_gesture_label = classifier.classify_static(d)

                    # # Dynamic
                    #     dynamic_gesture_id, dynamic_gesture_label = classifier.classify_dynamic(pre_processed_landmark_history_list)

        # Gesture handling
                    # self.handle_gestures(static_gesture_id, dynamic_gesture_id)

        # Image Drawing
                    debug_image = self.draw_bounding_rect(use_brect, debug_image,self.brect)

                    if self.show_information:
                        mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        debug_image = self.draw_hand_info(debug_image, self.brect, static_gesture_label)

            # else:
            #     if self.training and self.record:
            #         landmark_history.append(empty_landmarks)
            #     elif not self.training:
            #         landmark_history.append(empty_landmarks)

            if self.show_information:
                debug_image = self.draw_window_info(debug_image, win_dims, fps, dynamic_gesture_label)
            cv.imshow(winname, debug_image)

    # Shutdown
        cv.destroyAllWindows()
        vs.stop()
        tracker.stop()
        print("ScarletWitch Session Ended")

    # Reset stdout
        sys.stdout = stdout

    def terminate(self):
        self.running = False

    # def draw_hand_info(self, image, brect, handedness, static_gesture_text):
    def draw_hand_info(self, image, brect, static_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)

        # info_text = handedness.classification[0].label[0:]
        info_text = ""
        if static_gesture_text != "":
            info_text = info_text + ': ' + static_gesture_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        return image
    
    def draw_window_info(self, image, dimensions, fps, dynamic_gesture_text):
        # FPS
        cv.putText(image, "FPS:" + str(fps), (10, 35), cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        if self.training:
            # Training
            training_text = ["Off", "Static", "Dynamic"]
            cv.putText(image, "Training: " + training_text[self.training_mode], (10, 70),
                    cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

            # Label id
            cv.putText(image, "Label ID: " + str(self.label_id), (10, 110),
                    cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

            # Recording dynamic gesture
            if self.record:
                cv.putText(image, "RECORDING", (10, 145),
                        cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1, cv.LINE_AA)

            # Key Press
            cv.putText(image, self.last_key, (dimensions[0] - 35, 35),
                    cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)
        else:
            # Control mode
            mode_text = "View" if self.mode_view else "Movement"
            cv.putText(image, "Control Mode: " + mode_text, (10, 70),
                    cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

            # Dynamic gesture
            cv.putText(image, "Dynamic Gesture: " + dynamic_gesture_text, (10, 105),
                    cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (0, 0, 0), 1)

        return image

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # Wrist 1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # Wrist 2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # Thumb: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # Thumb: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # Thumb: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # Index finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # Index finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # Index finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # Index finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # Middle finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # Middle finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # Middle finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # Middle finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # Ring finger: Root
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # Ring finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # Ring finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # Ring finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # Little finger: base
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # Little finger: 2nd joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # Little finger: 1st joint
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # Little finger: fingertip
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

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

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

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
            return n / max_value if max_value > 0 else 0

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def pre_process_landmark_history(self, landmark_history):
        temp_landmark_history = list()

        for index in range(len(landmark_history)):
            landmarks = self.pre_process_landmark(landmark_history[index])
            temp_landmark_history += landmarks

        return temp_landmark_history
    
    # Wrapper for all geture controls
    def handle_gestures(self, static_id, dynamic_id):

        if not self.hand_pos_tracking and static_id == 1:  # Closed hand
            self.hand_pos_tracking = True
        elif self.hand_pos_tracking and static_id != 1:
            self.hand_pos_tracking = False

        if not self.hand_pos_tracking:
            self.prev = self.get_hand_pos()
        else:
            self.curr = self.get_hand_pos()
            self.handle_hand_pos(self.curr, self.prev)
            self.prev = self.curr

        if dynamic_id == 1:
            self.mode_view = not self.mode_view

    def handle_hand_pos(self, cur_pos, prev_pos):
        delta_x = cur_pos[0] - prev_pos[0]
        delta_y = cur_pos[1] - prev_pos[1]
        pyautogui.move(delta_x, delta_y)

    def get_hand_pos(self):
        x = (self.brect[0] + self.brect[2])/2
        y = (self.brect[1] + self.brect[3])/2
        hand_center = (round(x), round(y))

        return hand_center

    def get_data_length(self):
        datafilename = 'model/dynamic_classifier/dynamic_data.csv'

        with open(datafilename, 'r') as csv:
            first_line = csv.readline()

        ncol = first_line.count(',') + 1

        return ncol

    def log_csv(self, number, data):
        # Static gesture data
        if self.training_mode == 1:
            csv_path = 'model/static_classifier/static_data.csv'

        # Dynamic gesture data
        else:
            csv_path = 'model/dynamic_classifier/dynamic_data.csv'
        
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *data])
        
        return