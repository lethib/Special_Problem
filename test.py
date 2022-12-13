import cv2
import numpy as np

def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=2.0, skip_amount=30):
    # Don't affect original image
    image = image.copy()
    
    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=2)
    
    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])), 2)
    flow_end = (optical_flow_image[flow_start[:,:,1],flow_start[:,:,0],:1]*3 + flow_start).astype(np.int32)
    

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0
    
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y,x]), 
                        pt2=tuple(flow_end[y,x]),
                        color=(0, 255, 0), 
                        thickness=1, 
                        tipLength=.2)
    return image



# Load the image
img = cv2.imread("edges_frames/1.jpg")
next_img = cv2.imread("edges_frames/14.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
next_img_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

# Calculate the "good" features to track
features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Convert the features to an integer array
features = np.int0(features)

# Calculate the dense optical flow for the image
flow = cv2.calcOpticalFlowFarneback(gray, next_img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Loop through the features and draw circles on the image
for feature in features:
    # Get the x and y coordinates of the feature
    x, y = feature.ravel()
    
    # Get the flow vector for this feature
    dx, dy = flow[y, x]
    
    # Draw the arrow on the image
    cv2.arrowedLine(img, (int(x), int(y)), (int(x + dx), int(y + dy)), (0, 0, 255), thickness=1, tipLength=0.3)

# Display the image
cv2.imshow("Image with optical flow", img)
cv2.waitKey(0)
