import keyboard
import pyautogui
import sys
import time

from HandTracker import HandTracker

class ScarletWitch:
    def __init__(self, arguments, stdout=sys.stdout):
        self.running = False
        self.position_tracking = False
        self.args = arguments
        self.stdout = stdout

        keyboard.add_hotkey('.', self.terminate)

    def run(self):
        from threading import Thread

        # initialize collection timer
        ct = Timer()
        ct.run()

        # initialize handtracker and run thread
        ht = HandTracker(self.args, False)
        th = Thread(target=ht.run, name='handtracker')
        th.start()

        self.running = True

        prev_pos = (0, 0)

        # set stdout
        stdout = sys.stdout
        sys.stdout = self.stdout

        training = self.args.use_training_mode

        while not training and self.running:
            if ct.elapsed() > 1/60:

                if not self.position_tracking:
                    prev_pos = ht.get_hand_pos()

                self.position_tracking = ht.get_tracking()

                if self.position_tracking:
                    cur_pos = ht.get_hand_pos()
                    self.handle_hand_pos(cur_pos, prev_pos)
                    prev_pos = cur_pos

                ct.run()

        # reset stdout
        sys.stdout = stdout

        ht.terminate()

        # wait for handtracker to shut down
        th.join()

    def run_live(self):
        pass

    def run_training(self):
        pass

    def terminate(self):
        self.running = False

    def handle_hand_pos(self, cur_pos, prev_pos):
        delta_x = cur_pos[0] - prev_pos[0]
        delta_y = cur_pos[1] - prev_pos[1]
        pyautogui.move(delta_x, delta_y)


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