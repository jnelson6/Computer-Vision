
## Julia Nelson 
"""
k-means Segmentation
	Apply k-means segmentation on white-tower.png with k=10. 
	The distance function should only consider the RGB color channels and ignore pixel coordinates. 
	Randomly pick 10 RGB triplets from the existing pixels as initial seeds and run to convergence.  
	After k-means has converged, represent each cluster with the average RGB value of its members.



SLIC
Apply a variant of the SLIC algorithm to wt_slic.png, by implementing the following steps: 
	1. Divide the image in blocks of 50×50 pixels and initialize a centroid at the center of each block. 
	2. Compute the magnitude of the gradient in each of the RGB channels and use the square root 
	    of the sum of squares of the three magnitudes as the combined gradient magnitude. Move the centroids 
	    to the position with the smallest gradient magnitude in 3×3 windows centered on the initial centroids. 
	3. Apply k-means in the 5D space of x, y, R, G, B. Use the Euclidean distance in this space, 
	    but divide x and y by 2. 
	4. After convergence, display the output image: color pixels that touch two different clusters black 
	    and the remaining pixels by the average RGB value of their cluster. 


CAN USE 
image reading and writing functions
plotting functions
 can convert the images to a different format for reading them.


CANNOT USE 
filtering 
edge detection 
any other image processing functions. 

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
    new_filename = 'kmeans_out/{}_k{}_i{}.jpg'.format(filename, k, iter)

    bgr_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(new_filename, bgr_image)
