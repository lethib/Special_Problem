import numpy as np
import cv2 as cv

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
        cv.arrowedLine(image,
                        pt1=tuple(flow_start[y,x]), 
                        pt2=tuple(flow_end[y,x]),
                        color=(0, 255, 0), 
                        thickness=1, 
                        tipLength=.2)
    return image


cap = cv.VideoCapture('/Users/thibault/Documents/Georgia Tech/GTL/Special_Problem/new_method/output_video.mp4')
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while cap.isOpened():
    ret, frame2 = cap.read()
    print(ret)
    if not ret:
        print('No frames grabbed!')
        break

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    arrowed_img = put_optical_flow_arrows_on_image(frame2, flow)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', arrowed_img)
    k = cv.waitKey(5) & 0xFF
    if k == ord('q'):
        break
    
    cv.imwrite('opticalfb.png', frame2)
    cv.imwrite('opticalhsv.png', bgr)
    cv.imwrite('optical_arrow.png', arrowed_img)
    prvs = next

cap.release()
cv.destroyAllWindows()

if __name__ == '__main__':
    cap = cv.VideoCapture()
    while cap.isOpened():
        ret, frame = cap.read()
        cv.imshow("Image", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()