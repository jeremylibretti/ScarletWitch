# Get landmarks
# Process landmarks

import copy

import cv2 as cv
import mediapipe as mp

from threading import Thread

class HandTracker:
    def __init__(self, arguments, stream):
        self.mp_hands = mp.solutions.hands

        self.args = arguments
        self.stream = stream
        self.running = False

        # use_static_image_mode = self.args.use_static_image_mode
        use_static_image_mode = False
        min_detection_confidence = self.args.min_detection_confidence
        min_tracking_confidence = self.args.min_tracking_confidence

        self.hands = self.mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Run first iteration of image processing
        self.image = self.stream.read()
        self.image = cv.flip(self.image, 1)  # Mirror display
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        self.image.flags.writeable = False
        self.output = self.hands.process(self.image)
        self.image.flags.writeable = True

    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.run, args=()).start()
        return self

    def run(self):
        self.running = True

        while self.running:
            # Camera capture
            self.image = self.stream.read()
            self.image = cv.flip(self.image, 1)  # Mirror display

            # Hand detection
            self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

            self.image.flags.writeable = False
            results = self.hands.process(self.image)
            self.image.flags.writeable = True

            self.output = results

    def read(self):
        # return the frame most recently read
        return self.output

    def stop(self):
        # indicate that the thread should be stopped
        self.running = False