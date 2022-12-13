from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.metrics import r2_score

import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(array):
    rows, cols = array.shape
    I = []
    J = []
    for i in range(rows):
        for j in range(cols):
            if array[i,j] == 255:
                I.append(i)
                J.append(j)
    return I,J

def PolynomialRegression(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


def plot_validation_curve(model, X, y) -> None:
    degrees = np.arange(0,10)

    train_score, val_score = validation_curve(model, X[:, None], y, param_name='polynomialfeatures__degree', param_range=degrees)
    plt.plot(degrees, np.median(train_score, 1), color='b', label='training score') 
    plt.plot(degrees, np.median(val_score, 1), color='r', label='validation score') 
    plt.legend(loc="best")
    plt.ylim(0,1)
    plt.xlabel("degree of fit")
    plt.ylabel("score")
    plt.title("Validation curve")
    plt.show()

def get_best_degree(model, X, y) -> int:
    param_grid = {
        'polynomialfeatures__degree': np.arange(0,10),
        'linearregression__fit_intercept': [True, False]
    }

    grid = GridSearchCV(model, param_grid)
    grid.fit(X[:, None], y)
    return grid.best_params_["polynomialfeatures__degree"]

def fit_equation(degree, X, y, X_test):
    model = PolynomialRegression(degree)
    model.fit(X[:, None], y)
    y_pred = model.predict(X_test[:, None])
    r2 = r2_score(y, y_pred[:len(y)])
    return y_pred, r2

if __name__ == '__main__':
    print("Hello World !")