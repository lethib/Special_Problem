from functions import *

# Sample creation
sample = Sample(8, 2)

# Image Pixels
img_scrapped, shape = get_rgb_values('img_res/sobel.png')

# Loop on all the pixels
for i in range(len(img_scrapped) - 1):
    sample.complete_sample_init(img_scrapped[i], img_scrapped[i+1])
    rgb_values = np.zeros((2,4))
    for j in range(len(img_scrapped[0])):
        for k in range(len(rgb_values[0])):
            rgb_values[0,k] = delta_rgb(sample.table[0][2*k], sample.table[0][2*k+1])
            rgb_values[1,k] = delta_rgb(sample.table[1][2*k], sample.table[1][2*k+1])
        print(rgb_values)