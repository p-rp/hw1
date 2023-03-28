import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d

def main():
    
    # load the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        #subsample image 
        im_subsample = im_subsample[::2, ::2, :]
        plt.subplot(2, N_levels, i+1)
        plt.imshow(im_subsample)
        plt.axis('off')

    # subsampling without aliasing, visualize results on 2nd row
    im_subsample2 = im.copy()
    gaussian_filter = gaussian_kernel(5, 1)
    
    for i in range(N_levels):
        # Apply gaussian filter on each channel
        im_subsample2[:, :, 0] = filter2d(im_subsample2[:, :, 0], gaussian_filter)
        im_subsample2[:, :, 1] = filter2d(im_subsample2[:, :, 1], gaussian_filter)
        im_subsample2[:, :, 2] = filter2d(im_subsample2[:, :, 2], gaussian_filter)
        #subsample image 
        im_subsample2 = im_subsample2[::2, ::2, :]
        plt.subplot(2, N_levels, i+6)
        plt.imshow(im_subsample2)
        plt.axis('off')


    #### YOUR CODE HERE
    plt.show()
    #### END YOUR CODE
    
if __name__ == "__main__":
    main()
