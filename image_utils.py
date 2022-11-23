import cv2 as cv
import numpy as np
import random
# import matplotlib.pyplot as plt


def threshold_grayscale_image(img):
    """
    Thresholds a grayscale image into a mask
    """
    img = (np.asarray(img)).astype(np.uint8)
    # Threshold existing grayscale image and convert to RGB (3 channels)
    _, mask = cv.threshold(img, thresh=127, maxval=255, type=cv.THRESH_BINARY)
    return mask


def convert_image_from_gray_to_color_by_rgb_triple(img, rgb_color):
    mask = threshold_grayscale_image(img)
    # Convert mask to "RGB" (I belive all entries are (255,255,255) or (0,0,0))
    rgb_img = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    # Set all white pixels to the current color
    rgb_img[np.all(rgb_img == (255, 255, 255), axis=-1)] = rgb_color
    # plt.imshow(bgr_img)
    return rgb_img


def convert_image_from_gray_to_color_by_name(img, color_name):
    """
    Takes a grayscale image and the string name of a color and returns a colored rgb version of the image.
    """
    rgb_color = convert_color_name_to_rgb_triple(color_name)
    return convert_image_from_gray_to_color_by_rgb_triple(img, rgb_color)


# TODO: determine if we can parameterize this coloring scheme more / commit to a few concrete schemes
def convert_image_from_gray_to_color_by_idx(img, idx):
    """
    Converts the white pixels of an image to a specific color defined by the idx of the color

    COLOR_MAP taken from https://github.com/clovaai/rebias

    :param img: image pixel matrix
    :param idx: int index defining which color to use
    :return:
    """
    # Defines the color for up to 10 unique classes
    COLOR_MAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [225, 225, 0], [225, 0, 225],
                  [0, 255, 255], [255, 128, 0], [255, 0, 128], [128, 0, 255], [128, 128, 128]]
    rgb_color = COLOR_MAP[idx]
    return convert_image_from_gray_to_color_by_rgb_triple(img, rgb_color)


def convert_color_name_to_rgb_triple(color_name):
    """
    Takes the english name of a color and returns the rgb triple
    """

    color_dict = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'orange': (255, 128, 0)
    }
    if color_name in color_dict:
        return list(color_dict[color_name])
    else:
        raise ValueError(f'Illegal color name {color_name}. Choose one from {color_dict.keys()}')


# reference: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def add_sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output