from operator import truediv
from os import stat
import sys
import copy

import mediapipe as mp
import numpy as np
import cv2 as cv

from utils import CvFpsCalc
from handtracker import HandTracker
from gestureclassifier import GestureClassifier
from collections import deque
import webcam

import pyautogui

class ScarletWitch:
    def __init__(self, arguments, stdout=sys.stdout):
        self.args = arguments
        self.stdout = stdout

        self.cap_device = arguments.device
        self.cap_width = arguments.width
        self.cap_height = arguments.height

        self.show_info = True
        self.mode_view = True
        self.last_key = ""

        self.brect = [0, 0, 0, 0]
        self.use_brect = True

        self.freq = 57 # Num of time frames in a dynamic gesture

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.running = False

        self.position_tracking = False
        self.control_mode = False # False: Camera, True: Body
        self.current_position = (0, 0)
        self.previous_position = (0, 0)
        self.hand_center = (0, 0)
        self.tracking_hand = False

    def run(self):
        self.running = True

        # Set stdout
        stdout = sys.stdout
        sys.stdout = self.stdout

        # Video stream from webcam
        vs = webcam.MacVideoStream(self.cap_width, self.cap_height, self.cap_device).start()

        # Framerate calculator
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Webcam display
        winname = "Scarlet Witch"

        dims = [[640, 360], [720, 480], [1280, 720], [1920, 1080]]
        win_dims = dims[2]

        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.resizeWindow(winname, win_dims[0], win_dims[1])
        cv.moveWindow(winname, 0, 0)

        # Empty frame of hand landmarks
        empty_d = self.create_empty_data()

        # Coordinate history
        history_length = self.freq
        landmark_history = deque([empty_d for x in range(history_length)], maxlen=history_length)

        # Handtracking and gesture estimation threads
        tracker = HandTracker(self.args, vs).start()
        classifier = GestureClassifier(history_length)

        # Variable initialization
        dynamic_gesture_id = -1
        dynamic_gesture_label = ""
        dynamic_gesture = [dynamic_gesture_id, dynamic_gesture_label]

        # Main Loop
        while self.running:
            # Process keyb input
            key = cv.waitKey(10)

            self.key_action(key)

            # Camera capture
            image = vs.read()
            tracking_results = tracker.read()

            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Get fps
            fps = cvFpsCalc.get()

            # Landmark processing
            if tracking_results.multi_hand_landmarks is not None:
                for hand_landmarks in tracking_results.multi_hand_landmarks:

                    landmark_history, static_gesture, dynamic_gesture = self.process_landmarks(
                        debug_image, hand_landmarks, landmark_history, classifier)

                self.tracking_hand = True

            else:
                landmark_history.append(empty_d)
                self.tracking_hand = False

            # Gesture/control
            if self.position_tracking:
                # self.current_position = self.previous_position

                if self.tracking_hand:
                    landmarks = tracking_results.multi_hand_landmarks[0]
                    brect = self.calc_bounding_rect(debug_image, landmarks)
                    self.hand_center = ((brect[0] + brect[2])/2, (brect[1] + brect[3])/2)
                    self.current_position = self.hand_center

                    cur_pos = self.current_position
                    prev_pos = self.previous_position

                    delta_x = cur_pos[0] - prev_pos[0]
                    delta_y = cur_pos[1] - prev_pos[1]

                    # delta_depth = self.get_change_in_distance()

                    if static_gesture[0] == 1: # Hand closed
                        m = 5
                        pyautogui.move(delta_x*m, delta_y*m)

                    if static_gesture[0] == 2: # Pointer
                        if not walking:
                            walking = True
                            pyautogui.keyDown('w')
                    else:
                        walking = False
                        pyautogui.keyUp('w')

                self.previous_position = self.current_position


            if self.show_info:
                debug_image = self.draw_window_info(debug_image, win_dims, fps, dynamic_gesture[1])
            cv.imshow(winname, debug_image)

        # Shutdown
        cv.destroyAllWindows()
        vs.stop()
        tracker.stop()
        print("ScarletWitch Session Ended")

        # Reset stdout
        sys.stdout = stdout

    def create_empty_data(self):
        joint = np.zeros((21, 4))

        # Compute angles between joints
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
        v = v2 - v1 # [20, 3]
        # Normalize v
        # v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        angle = np.degrees(angle) # Convert radian to degree

        d = np.concatenate([joint.flatten(), angle])

        return d

    def key_action(self, key):
        if key > -1:
                self.last_key = chr(key)
        else:
            self.last_key = ""

        if key == 27:  # ESC
            self.last_key = "ESC"
            self.terminate()

        if key == 112:  # P
            self.position_tracking = not self.position_tracking
            print(f"Position tracking set to {str(self.position_tracking).upper()}\n")

        if key == 99: # C
            self.control_mode = not self.control_mode
            control = "Body"
            if self.control_mode:
                control = "Camera"
            print(f"Control mode switch to {control}\n")

    def process_landmarks(self, debug_img, landmarks, history, classifier):
        # Bounding box calculation
        self.brect = self.calc_bounding_rect(debug_img, landmarks)

        # Joint angle calculations
        landmark_data = self.landmarks_to_angles(landmarks)
        history.append(landmark_data)
    
        static_gesture_id, static_gesture_label = classifier.classify_static(landmark_data)
        dynamic_gesture_id, dynamic_gesture_label = classifier.classify_dynamic(history)

        # Image Drawing
        debug_image = self.draw_bounding_rect(debug_img, self.brect)

        self.mp_drawing.draw_landmarks(debug_image, landmarks, self.mp_hands.HAND_CONNECTIONS)
        debug_image = self.draw_hand_info(debug_image, self.brect, static_gesture_label)

        return history, [static_gesture_id, static_gesture_label], [dynamic_gesture_id, dynamic_gesture_label]

    def landmarks_to_angles(self, hand_landmarks):
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

        return d

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

    def draw_bounding_rect(self, image, brect):
        if self.use_brect:
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

    def draw_hand_info(self, image, brect, static_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)

        info_text = ""
        if static_gesture_text != "":
            info_text = static_gesture_text

        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        return image
    
    def draw_window_info(self, image, dimensions, fps, dynamic_gesture_text):
        # FPS
        cv.putText(image, "FPS:" + str(fps), (10, 35), cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Control toggle
        mode_text = "On" if self.mode_view else "Off"
        cv.putText(image, "Control Toggle: " + mode_text, (10, 70),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        # Dynamic gesture
        cv.putText(image, "Dynamic Gesture: " + dynamic_gesture_text, (10, 105),
                cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1, cv.LINE_AA)

        return image

    def terminate(self):
            self.running = False