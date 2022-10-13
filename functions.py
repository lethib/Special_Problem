import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Sample:
    
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.table = []
        self.create_sample()

    def create_sample(self):
        self.table = [[] for i in range(self.height)]
        #print(self.table)
        return self.table
    
    def add_pxl(self,pixel_up, pixel_down):
        """Add 2 pixels at the right of the sample in the same column"""
        # List circulaire
        if len(self.table[0]) >= self.width:
            self.table[0] = self.table[0][-(self.width-1):]
            self.table[1] = self.table[1][-(self.width-1):]

        self.table[0].append(pixel_up)
        self.table[1].append(pixel_down)

    def complete_sample_init(self, row_up, row_down):
        self.table[0].clear()
        self.table[1].clear()
        for i in range(self.width):
            self.table[0].append(row_up[i])
            self.table[1].append(row_down[i])
    
    def get_delta_values(self):
        rgb_values = np.zeros((2,int(self.width/2)))
        for k in range(len(rgb_values[0])):
            rgb_values[0,k] = delta_rgb(self.table[0][2*k], self.table[0][2*k+1])
            rgb_values[1,k] = delta_rgb(self.table[1][2*k], self.table[1][2*k+1])
        return rgb_values


def get_rgb_values(img_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    img = Image.open(img_path, 'r')
    width, height = img.size
    pxl_val = list(img.getdata())
    pxl_val = np.array(pxl_val).reshape(height,width,4) # 4 because it's RGBA (oppacity)
    #print(width, height)
    return pxl_val, pxl_val.shape

def delta_rgb(pix_1, pix_2):
    """Returns the mean RGB value differences between 2 pixels."""
    r1,g1,v1 = pix_1[0], pix_1[1], pix_1[2]
    r2,g2,v2 = pix_2[0], pix_2[1], pix_2[2]
    delta = np.sqrt((r2-r1)**2 + (g2-g1)**2 + (v2-v1)**2)
    return delta

if __name__ == '__main__':
    sample = Sample(8,2)
    img_scrp, shape = get_rgb_values('img_res/sobel.png')
    diff_val = [[] for i in range(shape[0])]
    # img_scrp[0] length = 82
    idx_diff = 0
    for i in range(0, len(img_scrp)-1, sample.height):
        sample.complete_sample_init(img_scrp[i], img_scrp[i+1])
        #print("***")
        #print(np.array(sample.table))
        for j in range(sample.width, len(img_scrp[0])):
            delta_val = sample.get_delta_values()
            mean_j = np.mean(delta_val)
            #print(f"*** Delta values for {j}:")
            #print(delta_val, mean_j)
            sample.add_pxl(img_scrp[i,j], img_scrp[i+1,j])
            delta_val = sample.get_delta_values()
            mean_jpp = np.mean(delta_val)
            #print(f"*** Delta values for {j+1}:")
            #print(delta_val, mean_jpp)
            diff_val[idx_diff].append(abs(mean_jpp - mean_j))
            #print(f"*** -> Diff mean values : {diff_val}")
        idx_diff += 1
    
    # We take the max value position
    max_val_pos = {}
    for row, list_val in enumerate(diff_val):
        if len(list_val) == 0: #Check is there is any empty list
            continue
        maxi = max(list_val)
        max_val_pos[shape[0] - row*2] = list_val.index(maxi)
        #print(max_val_pos)

    print(max_val_pos)

    img = mpimg.imread('img_res/sobel.png')
    plt.imshow(img, extent=[0, len(img_scrp[0]), 0, len(img_scrp)])
    plt.scatter(max_val_pos.values(), max_val_pos.keys(), c='r')
    plt.xlim(0, len(img_scrp[0]))
    plt.show()

    # print("*** Comparaison")
    # print(np.array(img_scrp[0][:8]))
    # print("\n")
    # print(np.array(img_scrp[1][:8]))
    # print("\n")
    # print(np.array(img_scrp[2][:8]))
    # print("\n")
    # print(np.array(img_scrp[3][:8]))
