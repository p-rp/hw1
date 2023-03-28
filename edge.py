import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y

def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)

    ### YOUR CODE HERE

    # Smooth image with Gaussian kernel
    img = filter2d(img, gaussian_kernel(10, 2))

    # Compute x and y derivate on smoothed image
    dx = partial_x(img)
    dy = partial_y(img)

    # Compute gradient magnitude
    # gradient_magnitude = np.sqrt(np.add(np.square(dx), np.square(dy)))  
    gradient_magnitude = np.hypot(dx, dy)       # sqrt(partial_x^2 + partial_y^2)

    # Visualize results
    rows = 2
    columns = 2

    # gradient image on x direction
    plt.subplot(rows, columns, 1)
    plt.imshow(dx, cmap=plt.cm.gray)
    plt.title('x-direction')
    plt.axis('off')

    # gradient image on y direction
    plt.subplot(rows, columns, 2)
    plt.imshow(dy, cmap=plt.cm.gray)
    plt.title('y-direction')
    plt.axis('off')

    # gradient magnitude
    plt.subplot(rows, columns, 3)
    plt.imshow(gradient_magnitude, cmap=plt.cm.gray)
    plt.title('Gradient Magnitude')
    plt.axis('off')

    plt.imshow(gradient_magnitude, cmap=plt.cm.gray)
    plt.axis('off')

    plt.show()
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()

