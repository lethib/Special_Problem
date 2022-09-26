from functions import *

# Sample creation
sample = Sample(8, 2)

# Image Pixels
img_scrapped, shape = get_rgb_values('img_res/sobel.png')

# Loop on all the pixels
for i in range(len(img_scrapped) - 1):
    sample.complete_sample_init(img_scrapped[i], img_scrapped[i+1])
    for j in range(8,len(img_scrapped[0])):
        rgb_values = sample.get_delta_values()
        sample.add_pxl(img_scrapped[i,j], img_scrapped[i+1,j])

print(rgb_values)