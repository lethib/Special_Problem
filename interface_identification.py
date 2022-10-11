from functions import *
import matplotlib.pyplot as plt

# Sample creation
sample = Sample(8, 2)

# Image Pixels
img_scrapped, shape = get_rgb_values('img_res/sobel.png')

# Loop on all the pixels
diff_values = [[] for i in range(shape[0])]
for i in range(len(img_scrapped) - 1):
    sample.complete_sample_init(img_scrapped[i], img_scrapped[i+1])
    for j in range(8,len(img_scrapped[0])):
        rgb_values = sample.get_delta_values()
        mean_j = np.mean(rgb_values)
        sample.add_pxl(img_scrapped[i,j], img_scrapped[i+1,j])
        rgb_values = sample.get_delta_values()
        mean_jpp = np.mean(rgb_values)
        diff_values[i].append(abs(mean_jpp - mean_j))


# We take the max value position
max_val_pos = {}
for row, list_val in enumerate(diff_values):
    if len(list_val) == 0:
        continue
    maxi = max(list_val)
    max_val_pos[shape[0] - row] = list_val.index(maxi)

plt.scatter(max_val_pos.values(), max_val_pos.keys())
plt.show()