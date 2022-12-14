# Semi-automated Identification and Tracking of moving interfaces

## The Project 

The purpose of this project is to determine the reduced mobility of Al grain boundaries. Grain boundaries were recorded in annealing experiments at different temperatures. We pursue to identify and track these interfaces to determine its velocity in the normal direction. To this aim, we will utilize **Computer Vision** and **Machine Learning**. 

Since the velocity of the grain boundary differs along the interface, the shape of it must be unequivocally identified. From the shape, the normal velocity will be calculated and thus the reduced mobility.

## The state of the project

Two methods were used for this project. The first one, the [**Sobel Pixel Intensity**](#sobel-pixel-intensity) was my first approach which did not work well. The second one, the [**Polynomial Regression and Dense Optical flow**](#polynomial-regression-and-dense-optical-flow) is my final solution for this problem. This metho deals with Machine Learning and Computer Vision. 

### Sobel Pixel Intensity
Before getting a good results, I have tried a first method base on the **Sobel Edge Detection** method. The interface was quite well identified by the method however, it was hard to compute the identification. 

The idea was to compare mean intensity of pixel's samples by :

- Creating 2 pixel's samples: one moved by 1 pixel in the x direction to the other
- Calculate the mean intensity of those 2 samples
- Compute the difference of mean intensity between those 2 samples

For each line of the image, we keep the maximum value of this difference (i.e. the higher contrast). 

However, this did not work well. The source still exists in the [`sobel_pxl_intensity`](https://github.gatech.edu/tmroz3/Special_Problem/tree/dev/src/sobel_pxl_intensity) folder. 

### Polynomial Regression and Dense Optical Flow

At this date (14/12/2022), the alogrithm works really well