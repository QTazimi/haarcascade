### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io


def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, n_channels).
    """

    out = None
    # YOUR CODE HERE
    out = io.imread(img_path) / 255

    return out


def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """

    # YOUR CODE HERE
    if len(image.shape) == 3:
        c = image.shape[2]
    else:
        c = 1

    h, w = image.shape[:2]
    print("Image height: %d" % h)
    print("Image width: %d" % w)
    print("Number of channels: %d" % c)
    return None


def crop(image, x1, y1, x2, y2):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        (x1, y1): the coordinator for the top-left point
        (x2, y2): the coordinator for the bottom-right point
        

    Returns:
        out: numpy array of shape(x2 - x1, y2 - y1, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image[y1:y2, x1:x2]
    # print(out.shape)

    return out


def resize(input_image, fx, fy):
    """Resize an image using the nearest neighbor method.
    Not allowed to call the matural function.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.
        fx (float): the resize scale on the original width.
        fy (float): the resize scale on the original height.

    Returns:
        np.ndarray: Resized image, with shape `(image_height * fy, image_width * fx, 3)`.
    """
    out = None

    ### YOUR CODE HERE
    h, w, c = input_image.shape[:3]
    nw, nh = int(fx * w), int(fy * h)
    out = np.zeros([nh, nw, c])

    for y in range(nh):
        for x in range(nw):
            out[y, x] = input_image[int(y / fy), int(x / fx)]
    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, divided by 255.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = factor * (image - 0.5) + 0.5

    return out


def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(image_height, image_width, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(image_height, image_width)`.
    """
    out = None
    # out = input_image[:, :, 0] / 3 + input_image[:, :, 1] / 3 + input_image[:, :, 2] / 3
    out = np.dot(input_image[..., :3], np.array([0.299, 0.587, 0.114]))
    return out


def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.
  
                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th
    
    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None
    h, w = grey_img.shape[:2]
    out = np.zeros([h, w])

    for y in range(h):
        for x in range(w):
            out[y, x] = 1 if grey_img[y][x] > th else 0
    return out


def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    out = None
    ### YOUR CODE HERE

    h, w = image.shape[:2]
    Kh, Kw = kernel.shape[:2]
    Ph, Pw = int(Kh / 2), int(Kw / 2)

    Ipad = np.zeros((h + Ph * 2, w + Pw * 2))
    Ipad[Ph:Ph + h, Pw:Pw + w] = image

    out = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            out[i][j] = np.sum(np.multiply(Ipad[i:i + Kh, j:j + Kw], kernel[::-1, ::-1]))
    return out


def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)
    #
    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE

    out = np.zeros(np.shape(image))

    if image.shape[2] == 3:
        for i in range(image.shape[2]):
            out[:, :, i] = conv2D(image[:, :, i], kernel)
    else:
        out = conv2D(image, kernel)

    return out


def test_cov(image):
    simKernel = np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ])
    # Run your conv_nested function on the test image
    return conv(image, simKernel)


def gauss2D(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    # 归一化
    return g / g.sum()


def test_gauss(image):
    return conv(image, gauss2D(125, 1.5))


def sobel(image, operator):
    operatorX = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
    )
    temp = operatorX.reshape(operatorX.size)
    temp = temp[::-1].reshape(operatorX.shape)
    operatorY = np.transpose(temp)[::-1]
    # [-1, -2, -1], [0, 0, 0], [1, 2, 1]

    if operator == "horizontal":
        operator = operatorX
    elif operator == "vertical":
        operator = operatorY
    else:
        print("Unknown operator: %s" % operator)
        return None
    return conv(image, operator)


def pyramid(image):
    K = gauss2D(10, 1)
    I1 = resize(conv(image, K), 0.5, 0.5)
    I2 = resize(conv(I1, K), 0.5, 0.5)
    I3 = resize(conv(I2, K), 0.5, 0.5)
    I4 = resize(conv(I3, K), 0.5, 0.5)

    return I1, I2, I3, I4


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE
    h, w = image.shape[:2]
    Kh, Kw = kernel.shape[:2]
    Ph, Pw = int(Kh / 2), int(Kw / 2)

    Ipad = np.zeros((h + Ph * 2, w + Pw * 2))
    Ipad[Ph:Ph + h, Pw:Pw + w] = image

    out = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            out[i][j] = np.sum(np.multiply(Ipad[i:i + Kh, j:j + Kw], kernel))
    return out


def corr3(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel

    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE

    out = np.zeros(np.shape(image))

    if image.shape[2] == 3:
        for i in range(image.shape[2]):
            out[:, :, i] = corr(image[:, :, i], kernel[:, :, i])
    else:
        out = corr(image, kernel)

    return out


def test_corr3(image):
    test_image = crop(image, 380, 165, 462, 220)
    print(image.shape, test_image.shape)
    out = corr3(image, test_image)
    return out
