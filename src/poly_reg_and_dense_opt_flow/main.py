from classes import *
from functions import *

### Path to content folders ###
original_frames = '/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/img/original/'
edges_frames = '/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/img/edges_detected/'
dense_flow_frames = '/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/img/dense_flow/'
video_folder = '/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/videos/'

### Compute edge detection for each frames
video = CreateVideos((400,320), 3)
video.edges_frames(original_frames + '*.jpg', edges_frames)

### Select good points on the first frame
first_frame = cv2.imread(edges_frames + '1.jpg')
good_points = select_points(first_frame, 20)

### Machine Learning - Polynomial Regression
X_test = np.arange(100, 270)
poly_reg = PolynomialRegression(good_points[:,0], good_points[:,1], X_test)
poly_reg.plot_validation_curve()
poly_reg.plot_regression(first_frame)

### Computer Vision - Dense Optical Flow
compute_dense_optical_flow(edges_frames + '*.jpg', dense_flow_frames, good_points)

### Create videos
video.video(original_frames + '*.jpg', video_folder + 'original.mp4')
video.video(dense_flow_frames + '*.jpg', video_folder + 'dense_flow.mp4')