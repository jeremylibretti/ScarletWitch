# Add to queue
# Write to file

import csv

from collections import Counter

from model import StaticClassifier
from model import DynamicClassifier

from collections import deque

class GestureClassifier:
    def __init__(self, history_length):

        # Read labels ###########################################################
        with open('model/static_classifier/static_classifier_labels.csv',
                encoding='utf-8-sig') as f:
            self.static_classifier_labels = csv.reader(f)
            self.static_classifier_labels = [
                row[0] for row in self.static_classifier_labels
            ]
        with open(
                'model/dynamic_classifier/dynamic_classifier_labels.csv',
                encoding='utf-8-sig') as f:
            self.dynamic_classifier_labels = csv.reader(f)
            self.dynamic_classifier_labels = [
                row[0] for row in self.dynamic_classifier_labels
            ]

        self.static_classifier = StaticClassifier()
        self.dynamic_classifier = DynamicClassifier(score_th=0.9)

        self.history_length = history_length
        self.dynamic_gesture_history = deque([0 for x in range(self.history_length)], maxlen=self.history_length)

    def classify_static(self, landmarks):
        ##### Static Gesture Calculation
        # Hand sign classification [0: Open, 1: Closed, 2: Pointer, 3: OK]
        static_gesture_id = self.static_classifier(landmarks)
        
        return static_gesture_id, self.static_classifier_labels[static_gesture_id]

    def classify_dynamic(self, landmark_history):
        ##### Dynamic Gesture Calculation
        # Dynamic gesture classification
        dynamic_gesture_id = 0
        classification = self.dynamic_classifier(landmark_history)
            
        # Calculates the gesture IDs in the latest detection
        self.dynamic_gesture_history.append(classification)
        if self.dynamic_gesture_history[-1] == self.dynamic_gesture_history[-2] == self.dynamic_gesture_history[-3]:
            dynamic_gesture_id = classification
        
        return dynamic_gesture_id, self.dynamic_classifier_labels[dynamic_gesture_id]

    def get_hand_position(self, debug_image, hand_landmarks):

        ##### Hand Position Calculation
        brect = self.calc_bounding_rect(debug_image, hand_landmarks)

        x = (brect[0] + brect[2])/2
        y = (brect[1] + brect[3])/2
        hand_center = (round(x), round(y))

        return hand_center