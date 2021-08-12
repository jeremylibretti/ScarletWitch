import keyboard
import pyautogui
import sys
import time

from HandTracker import HandTracker

class Timer:
    def __init__(self):
        self.init = time.time()
        self.start = time.time()
        self.running = False

    def run(self):
        self.start = time.time()
        self.running = True

    def stop(self):
        self.running = False

    def log(self):
        return f"{time.time() - self.init},"

    def elapsed(self):
        return time.time() - self.start if self.running else 0

class ScarletWitch:
    def __init__(self, stdout=sys.stdout):
        self.running = False
        self.position_tracking = False
        self.stdout = stdout

        keyboard.add_hotkey('.', self.terminate)

    def run(self):
        from threading import Thread

        # initialize collection timer
        ct = Timer()
        ct.run()

        # initialize handtracker and run thread
        ht = HandTracker(False, 0.7, 0.5)
        th = Thread(target=ht.run, name='webcam')
        th.start()

        self.running = True
        prev_pos = (0, 0)
        tracking = False

        # set stdout
        stdout = sys.stdout
        sys.stdout = self.stdout

        while self.running:
            if ct.elapsed() > 1/60:
                
                if not tracking:
                    prev_pos = ht.get_hand_pos()

                tracking = ht.get_tracking()

                if tracking:
                    cur_pos = ht.get_hand_pos()
                    delta_x = cur_pos[0] - prev_pos[0]
                    delta_y = cur_pos[1] - prev_pos[1]
                    pyautogui.move(delta_x, delta_y)
                    prev_pos = cur_pos

                ct.run()

        # reset stdout
        sys.stdout = stdout

        ht.terminate()

        # wait for handtracker to shut down
        th.join()

    def terminate(self):
        self.running = False