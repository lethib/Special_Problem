import cv2

def sobel_detec(img_path):
    """Edge Detection using the Sobel method"""
    img = cv2.imread(img_path)

    # Img blurred to have a better sobel detection
    img_blur = cv2.GaussianBlur(img, (3,3), sigmaX=0, sigmaY=0)

    sobel_y_blurr = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)

    return sobel_y_blurr

if __name__ == '__main__':
    res = sobel_detec('VideoAndFrames/1.jpg')
    cv2.imwrite('img_res/sobel_detec.jpg',res)