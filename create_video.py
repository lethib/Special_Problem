import cv2
import numpy as np
import glob
import re

frameSize = (400, 320)

out = cv2.VideoWriter('output_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 3, frameSize)
out1 = cv2.VideoWriter('original_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 3, frameSize)

files = glob.glob('/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/VideoAndFrames/*.jpg')

# Lexicographically order your input array of strings (e.g. respect numbers in strings while ordering)
files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

# Apply the Canny Edge Detection on all the frame of the video
i=1
for filename in files:
    img = cv2.imread(filename)
    out1.write(img)
    gaussian = cv2.GaussianBlur(img, (7,7), 0)
    edges_gaussian = cv2.Canny(gaussian, threshold1=20, threshold2=40)
    cv2.imwrite(f'/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/edges_frames/{i}.jpg', edges_gaussian)
    i += 1

out1.release()

files = glob.glob('/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/edges_frames/*.jpg')

# Lexicographically order your input array of strings (e.g. respect numbers in strings while ordering)
files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

for filename in files:
    img = cv2.imread(filename)
    out.write(img)

out.release()