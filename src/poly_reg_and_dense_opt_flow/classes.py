### SkLearn Import for Machine Learning Processing ###
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve, GridSearchCV

### Import Numpy and Matplotlib to handle and visualise the data ###
import numpy as np
import matplotlib.pyplot as plt

### Import OpenCv ###
import cv2

### Import useful modules ###
import glob
import re


### Classes ###

class PolynomialRegression():
    """
    #### Create a 2nd order Polynomial Regression Model to fit the interface equation.
    ---
    ### Parameters:

    - `x_coor` (list): list of the point's x-coordinates.
    - `y_coor` (list): list of the point's y-coordinates.
    - `x_test_coor` (list): list of the points used to plot the Polynomial Regression.
    ---
    ### Attributes:

    - `X` (list): list of the point's x-coordinates.
    - `y` (list): list of the point's y-coordinates.
    - `X_test` (list): list of the points used to plot the Polynomial Regression.
    - `model`: the model used.
    - `y_pred` (list): list of the point's y-coordinates predicted by the model.
    - `coef` (list): coefficients of the polynomial regression
    ---
    ### Methods: 

    - `plot_validation_curve`: plot the validation curve of the model.
    - `plot_regression`: plot the regression on the processed image.
    """

    def __init__(self, x_coor: list, y_coor: list, x_test_coor: list) -> None:
        self.X = x_coor
        self.y = y_coor
        self.X_test = x_test_coor
        self.model = self.create_model()
        self.y_pred, self.coef = self.fit_equation()

    def create_model(self):
        return make_pipeline(PolynomialFeatures(2), LinearRegression())
    
    def plot_validation_curve(self) -> None:
        degrees = np.arange(0,10)

        train_score, val_score = validation_curve(self.model, self.X[:, None], self.y, param_name='polynomialfeatures__degree', param_range=degrees)
        plt.plot(degrees, np.median(train_score, 1), color='b', label='training score') 
        plt.plot(degrees, np.median(val_score, 1), color='r', label='validation score') 
        plt.legend(loc="best")
        plt.ylim(0,1)
        plt.xlabel("degree of fit")
        plt.ylabel("score")
        plt.title("Validation curve")
        plt.show()
    
    def fit_equation(self):
        self.model.fit(self.X[:, None], self.y)
        y_pred = self.model.predict(self.X_test[:, None])
        coef = self.model[1].coef_
        return y_pred, coef

    def plot_regression(self, processed_img):
        plt.title(f"Polynomial Regression of 2 degree \n $y = {round(self.coef[2], 4)}x^2 + {round(self.coef[1], 4)}x + {round(self.coef[0], 4)}$")
        plt.imshow(processed_img)
        plt.plot(self.X_test, self.y_pred, color='r')
        plt.show()


class CreateVideos():
    """
    #### Create videos for visualisation. 
    ---
    ### Parmeters:

    - `vid_size` (tuple): the size of the video
    - `frame_per_sec` (int): the number of frame per seconds
    ---
    ### Attributes: 

    - `frame_size` (tuple): the size of the video
    - `frame_per_sec` (int): the number of frame per seconds
    ---
    ### Methods:

    - `video`: create a video from img
    - `edges_frames`: apply the Canny Edge Detection on all the frame of the original video
    """

    def __init__(self, vid_size: tuple, frame_per_sec: int) -> None:
        self.frame_size = vid_size
        self.frame_per_sec = frame_per_sec
    
    def video(self, input_img_path: str, output_vid_path: str) -> None:
        out = cv2.VideoWriter(output_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), self.frame_per_sec, self.frame_size)

        files = glob.glob(input_img_path)
        # Lexicographically order your input array of strings (e.g. respect numbers in strings while ordering)
        files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

        for filename in files:
            img = cv2.imread(filename)
            out.write(img)
        
        out.release()
    
    def edges_frames(self, input_img_path: str, output_img_path: str) -> None:
        files = glob.glob(input_img_path)
        # Lexicographically order your input array of strings (e.g. respect numbers in strings while ordering)
        files.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

        for i, filename in enumerate(files):
            img = cv2.imread(filename)
            gaussian = cv2.GaussianBlur(img, (7,7), 0)
            edges_gaussian = cv2.Canny(gaussian, threshold1=20, threshold2=40)
            cv2.imwrite(output_img_path + f'{i+1}.jpg', edges_gaussian)

if __name__ == '__main__':
    X = np.arange(1, 10)
    y = np.log(X)
    X_test = np.arange(1,15)
    reg = PolynomialRegression(X, y, X_test)
    img = cv2.imread('/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/content/img/edges_detected/1.jpg')
    reg.plot_validation_curve()
    print(len(reg.coef))