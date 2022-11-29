import cv2
from functions import *


### Edge Detection
# Read Img
img = cv2.imread('new_method/1.jpg')

# Blur the img
gaussian = cv2.GaussianBlur(img, (7,7), 0)

edges_gaussian = cv2.Canny(gaussian, threshold1=20, threshold2=40)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(edges_gaussian)
ax[1].set_title('Canny Edge Detection')
plt.tight_layout(pad=1.7)
plt.show()
###

### Convert the edge detection to a scatter plot
X, Y = scatter_plot(edges_gaussian)
###

### Select 3 points
plt.waitforbuttonpress()
while True:
    points = []
    while len(points) < 5:
        plt.title("Select 15 points with the mouse")
        plt.scatter(X, Y)
        points = np.asarray(plt.ginput(5, timeout=0)) #never times out

    plt.title("Key click to leave")

    plt.plot(points[:,0], points[:, 1], color='r')
    plt.draw()

    if plt.waitforbuttonpress():
        plt.close('all')
        break

plt.close('all')

# plot_validation_curve(PolynomialRegression(0), points[:,0], points[:,1])
# plt.close('all')
# deg = get_best_degree(PolynomialRegression(0), points[:,0], points[:,1])

X_test = np.arange(30, 130)

y_pred, r2 = fit_equation(2, points[:,0], points[:,1], X_test)

print(f"R^2 score: {r2}")

plt.title(f"Polynomial Regression of 2 degree - R^2 score: {r2}")
plt.scatter(X, Y)
plt.plot(X_test, y_pred, color='r')
plt.show()