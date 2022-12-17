from classes import *

def select_points(processed_img, nb_points: int) -> list:
    """
    #### Manually select the good points that corresponds to the interface.
    ---
    ### Parameters:

    - `processed_img`: the image which will be shown to click on it
    - `nb_points`: the number of points to click
    ---
    ### Output:

    - `points` (list): list of the points clicked
    """
    while True:
        points = []
        while len(points) < nb_points:
            plt.title(f"Select {nb_points} points with the mouse")
            plt.imshow(processed_img)
            points = np.asarray(plt.ginput(nb_points, timeout=0)) #never times out

        plt.title("Key click to leave")

        plt.plot(points[:,0], points[:, 1], color='r')
        plt.draw()

        if plt.waitforbuttonpress():
            plt.close('all')
            break

    plt.close('all')
    return points

def compute_dense_optical_flow(img_files: str, output_folder_path: str, points_to_track: list) -> None:
    """
    #### Compute the Dense Optical flow frame by frame.
    ---
    ### Parameters:

    - `img_files` (str): path to the image files
    - `output_folder_path` (str): path to the output folder
    - `points_to_track` (list): list of the points that will be track
    ---
    ### Output:

    `None`
    """
    files = glob.glob(img_files)
    # Lexicographically order your input array of strings (e.g. respect numbers in strings while ordering)
    files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

    for i in range(1, len(files)):
        img = cv2.imread(files[0])
        next_img = cv2.imread(files[i])

        # Convert images to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        next_img_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        # features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        # features = np.int0(features)

        features = np.int0(points_to_track)

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
        
        # Save the image
        cv2.imwrite(output_folder_path + f'1-{i}.jpg', img)

if __name__ == '__main__':
    img = cv2.imread('/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/img/edges_detected/1.jpg')
    points = select_points(img, 20)
    imgs = '/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/img/edges_detected/*.jpg'
    outputs = '/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/img/dense_flow/'
    compute_dense_optical_flow(imgs, outputs, points)
