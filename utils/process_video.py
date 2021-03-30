import cv2
import numpy as np
import os
import sys
import datetime

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from visualize_cv2 import model, display_instances, class_names

capture = cv2.VideoCapture('potholedrive.mp4')
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')

save_dir = os.path.join(os.getcwd(), "output")
if not os.path.exists(save_dir):
   os.makedirs(save_dir)

file_name = "videofile_maksed_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
file_name = os.path.join(save_dir, file_name)
output = cv2.VideoWriter(file_name, codec, 60.0, size)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
