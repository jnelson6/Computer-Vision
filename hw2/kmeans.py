
"""
Julia Nelson
CS 558
Homework 2 - Question 1 (k-means segmentation)
“I pledge my honor that I have abided by the Stevens Honor System.”

Question 1:  k-means Segmentation
	Apply k-means segmentation on white-tower.png with k=10. 
	The distance function should only consider the RGB color channels and ignore pixel coordinates. 
	Randomly pick 10 RGB triplets from the existing pixels as initial seeds and run to convergence.  
	After k-means has converged, represent each cluster with the average RGB value of its members.

CAN USE 
- image reading and writing functions
- plotting functions
- can convert the images to a different format for reading them.


CANNOT USE 
- filtering 
- edge detection 
- any other image processing functions. 

the complexity of k-means is linear with respect to the number of pixels and k. 
Start developing on small images with small values of k.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def kmeans_seg(filename, k, iter):
    # Read in the image
    image = cv2.imread('white-tower.png')

    # Change color to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plt.imshow(image)
    # plt.show()

    print(image.shape)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))
    print(pixel_vals.shape)

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    stop_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        iter,
        0.25)

    retval, labels, centers = cv2.kmeans(
        pixel_vals,
        k,
        None,
        stop_criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image
    


if __name__ == '__main__':
    filename = "white-tower.png"
    k = 10
    iter = 100

    segmented_image = kmeans_seg(filename, k, iter)

    # plt.imshow(segmented_image)

    filename = os.path.splitext(filename)[0]
    new_filename = '{}_k{}_i{}.jpg'.format(filename, k, iter)

    bgr_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(new_filename, bgr_image)
