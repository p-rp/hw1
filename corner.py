import numpy as np
from utils import filter2d, partial_x, partial_y
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    response = None
    
    ### YOUR CODE HERE
    # Compute partial derivatives 𝐼x and 𝐼y at each pixel
    Ix = partial_x(img)
    Iy = partial_y(img)
    
    Ix2 = np.square(Ix)
    Iy2 = np.square(Iy)
    IxIy = Ix * Iy

    # Compute (weighted) second moment matrix in a window around each pixel. Using box filter
    box_filter = np.ones((window_size, window_size)) / 9
    Ix2 = filter2d(Ix2, box_filter)
    Iy2 = filter2d(Iy2, box_filter)
    IxIy = filter2d(IxIy, box_filter)

    # Compute corner response function. R=Det(M)-k(Trace(M)^2)
    response = (Ix2*Iy2 - IxIy*IxIy) - k*np.square(Ix2 + Iy2)
    
    ### END YOUR CODE  

    return response

def main():
    img = imread('building.jpg', as_gray=True)

    ### YOUR CODE HERE 
    
    # Compute Harris corner response
    response = harris_corners(img)

    # Threshold on response
    threshold = response > 0.03

    # Perform non-max suppression by finding peak local maximum
    nms = peak_local_max(response, min_distance=6, threshold_abs=0.01)

    # Visualize results
    rows = 2
    columns = 2

    # corner response
    plt.subplot(rows, columns, 1)
    plt.imshow(response, cmap=plt.cm.gray)
    plt.title('Corner Response')
    plt.axis('off')

    # Threshold
    plt.subplot(rows, columns, 2)
    plt.imshow(threshold, cmap=plt.cm.gray)
    plt.title('Threshold')
    plt.axis('off')

    # Detected corners after non-maximum suppression
    plt.subplot(rows, columns, 3)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.autoscale(False)
    plt.plot(nms[:, 1], nms[:, 0], 'rx')
    plt.title('Peak Local Max')
    plt.axis('off')

    plt.show()
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()
