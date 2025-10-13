import os 
import sys 
import cv2 
import select

from contextlib import contextmanager

def detect_keyboard_press(key='q'):
    if select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip()
        if key in line:
            print(f"{key} has been pressed")
            return True
        else:
            return False

def select_one_from_list(options):
    for i, option in enumerate(options):
        print(f"{i}: {option}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 0 <= choice < len(options):
                return options[choice]
            else:
                print("Invalid number, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")


class VideoSaver(object):
    def __init__(self, filepath, vid_fps = 30, frame_size=(2048, 2048)):
        # Initialize video writer object
        self.video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'), vid_fps, frame_size)
    
    def write2video(self, frame):
        # Write the frame to the output files
        self.video.write(frame)
    
    def done(self):
        self.video.release()


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


