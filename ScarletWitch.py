import keyboard
import sys
import pyautogui
from HandTracker import HandTracker
import time

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
        self.stdout = stdout
        keyboard.add_hotkey('.', self.terminate)
        self.position_tracking = False

    def handle_args(self):
        log_mode = "log" in self.args

        if self.args:
            arg = self.args[0]
            if arg == "stop":
                return DWELL_STOP, log_mode
            elif arg == "move":
                return DWELL_MOVE, log_mode

        return DWELL_NOOP, log_mode

    def handle_hand_pos(self, pos, prev_pos, dt, config):
        global WKEY_UP
        dwell_action, log_mode = config

        if log_mode:
            print(dt.log(), f"{pos[0]}, {pos[1]}")

        # check if eye position has changed
        displace_left, displace_right = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
        moved = (displace_left != 0 and displace_right != 0)

        if not moved and not dt.running:
            dt.run()
        elif moved and dt.running:
            dt.stop()
            if dwell_action == DWELL_MOVE and not WKEY_UP:
                WKEY_UP = True
                pyautogui.keyUp('w')

        if dt.running and dt.elapsed() > 3:
            if dwell_action == DWELL_STOP:
                self.terminate()
                return
            elif dwell_action == DWELL_MOVE and WKEY_UP:
                WKEY_UP = False
                pyautogui.keyDown('w')

        if not log_mode and pos[0] == pos[1]:
            pyautogui.move(pos[0] * MOVE_SPEED, 0)

    def handle_static_gesture(self, gesture):
        pass

    def handle_dynamic_gesture(self, gesture):
        pass

    def handle_gesture(self, pos, tracking):
        if tracking:
            pyautogui.moveTo(pos[0][0], pos[0][1])
            

    def run(self, mode):
        from threading import Thread

        # config = self.handle_args()

        # initialize collection timer
        ct = Timer()
        ct.run()

        # initialize handtracker and run thread
        ht = HandTracker(False, 0.7, 0.5)
        th = Thread(target=ht.run, name='webcam')
        th.start()

        self.running = True
        # prev_pos = (0, 0)

        # set stdout
        stdout = sys.stdout
        sys.stdout = self.stdout

        # initialize dwelling timer
        dt = Timer()
        dt.run()

        while self.running:
            if ct.elapsed() > 1/60:
                pos = ht.get_hand_pos()
                sg = ht.get_static_gesture()
                dg = ht.get_dynamic_gesture()
                tracking = ht.get_tracking()

                if mode == "Print":
                    print("Pos: "+str(pos)+"\t SG: "+str(sg)+"\t DG: "+str(dg))
                elif mode == "Game":
                    print("Pos: "+str(pos)+"\t SG: "+str(sg)+"\t DG: "+str(dg))
                    self.handle_gesture(pos, tracking)

                # prev_pos = pos
                ct.run()

        # reset stdout
        sys.stdout = stdout

        ht.terminate()

        # wait for handtracker to shut down
        th.join()

    def terminate(self):
        self.running = False