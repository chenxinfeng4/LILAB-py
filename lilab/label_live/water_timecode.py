import cv2
import numpy as np

code_w, code_h = 120, 60
def water_timecode(frame:np.ndarray, code:int):
    frame[:code_h, :code_w] = 0
    if 0<=code<=999:
        strn = f'{code:03.0f}'
        cv2.putText(frame, strn, (5//2, 105//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 4//2, (255,255,255), 2)
