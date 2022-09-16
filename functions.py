import numpy as np
from PIL import Image

class Sample:
    
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

def get_rgb_values(img_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    img = Image.open(img_path, 'r')
    width, height = img.size
    pxl_val = list(img.getdata())
    pxl_val = np.array(pxl_val).reshape(height,width,4) # 4 because it's RGBA (oppacity)
    return pxl_val, pxl_val.shape

if __name__ == '__main__':
   res, shape = get_rgb_values('img_res/sobel.png')
   print(res[50], shape)