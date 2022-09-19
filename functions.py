import numpy as np
from PIL import Image

class Sample:
    
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.table = []
        self.create_sample()

    def create_sample(self):
        self.table = [[] for i in range(self.height)]
        print(self.table)
        return self.table
    
    def add_pxl(self,pixel_up, pixel_down):
        """Add 2 pixels at the right of the sample in the same column"""
        # List circulaire
        if len(self.table[0]) >= self.width:
            self.table[0] = self.table[0][-(self.width-1):]
            self.table[1] = self.table[1][-(self.width-1):]

        self.table[0].append(pixel_up)
        self.table[1].append(pixel_down)

    def complete_sample(self):
        for i in range(self.width-1):
            self.table[0].append(i)
            self.table[1].append(i+self.width)


def get_rgb_values(img_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    img = Image.open(img_path, 'r')
    width, height = img.size
    pxl_val = list(img.getdata())
    pxl_val = np.array(pxl_val).reshape(height,width,4) # 4 because it's RGBA (oppacity)
    print(width, height)
    return pxl_val, pxl_val.shape

def delta_rgb(pix_1, pix_2):
    """Returns the mean RGB value differences between 2 pixels."""
    r1,g1,v1 = pix_1[0], pix_1[1], pix_1[2]
    r2,g2,v2 = pix_2[0], pix_2[1], pix_2[2]
    delta = np.sqrt((r2-r1)**2 + (g2-g1)**2 + (v2-v1)**2)
    return delta

if __name__ == '__main__':
   res, shape = get_rgb_values('img_res/sobel.png')
   sample_test = Sample(8,2)
   sample_test.complete_sample()
   sample_test.add_pxl([1,2],[3,4])
   sample_test.add_pxl([45,67],[89,100])
   print(res[50], shape, sample_test.table)