import cv2 
from functions import *

## Edge Detection
# Read Img
img = cv2.imread('new_method/1.jpg')

# Blur the img 
gaussian = cv2.GaussianBlur(img, (7,7), 0)

# Canny Edge Detection
edges_gaussian = cv2.Canny(gaussian, threshold1=20, threshold2=40)

### Select points
plt.waitforbuttonpress()
while True:
    points = []
    while len(points) < 15:
        plt.title("Select 15 points with the mouse")
        plt.imshow(edges_gaussian)
        points = np.asarray(plt.ginput(15, timeout=0)) #never times out

    plt.title("Key click to leave")

    plt.plot(points[:,0], points[:, 1], color='r')
    plt.draw()

    if plt.waitforbuttonpress():
        plt.close('all')
        break

plt.close('all')