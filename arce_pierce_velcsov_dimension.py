""" arce_pierce_velcsov_dimension.py
Implementation of the proposed fractal dimension estimator in:
    Arce, W., Pierce, J. and Velcsov, M.T., 2021. A single-scale fractal feature for classification of color images: A
        virus case study. Chaos, Solitons & Fractals, 147, p.110849.

This method is a generalized form of the Minkowski-Bouligand, box-counting dimension, where epsilon is fixed at the
total number of pixels in the image.  This allows for extremely precise estimations for square fractal structures,
such as the Sierpinski Carpet, and for slightly more precise estimations for non-square fractal structures, such
as the Sierpinski Triangle.  To compensate for this loss in accuracy, an alpha mask is used to filter out unneeded
pixels, such as the background.

This source code fully implements the derived equations and has been validated on the Sierpinski Carpet.

Author:     Walker Arce
Version:    0.2
Date:       21 October 2020
"""
import cv2
import numpy as np
import math


def get_l_min(fractal, bit_depth):
    """

    :param fractal:
    :param bit_depth:
    :return:
    """
    return bit_depth / (fractal.shape[0] * fractal.shape[1])


def flatten_image(image):
    """
    Puts each color channel of image into a 1xn array to ease processing.
    :param image: list[ndarray]
        Image must first be split into color channels and is expected to be passed within a list.
    :return: list[ndarray]
        Each color channel is flattened and converted to a Numpy array.
    """
    flat_image = []
    for channel in image:
        flat_image.append(np.array(channel.flatten()))
    return flat_image


def alpha_filter(image, alpha_mask):
    """
    Removes all elements that are covered by alpha layer for each channel of the image.  Also strips the alpha channel
    from the image.
    :param image: list[ndarray]
        The color channels of the image, alpha channel must not be included.
    :param alpha_mask: ndarray
        The alpha channel of the image.
    :return: list[ndarray]
        The filtered channels of the image without the alpha channel.
    """
    channels = []
    for channel in image:
        channels.append(channel[np.where(alpha_mask == 255)])
    return channels


def channel_intensities(channels, bit_depth=255, invert=False):
    """
    Calculates the intensity for each channel of the image.
    :param channels: ndarray[list]
        The list of 1xn color channels to be processed.
    :param bit_depth: int (default: 255)
        The maximum possible value of the pixels.
    :param invert: bool (default: False)
        Setting to true will invert the pixels before processing.
    :return: list[float]
        Returns the intensities of each channel in the same order as they were received.
    """
    intensities = []
    if invert is True:
        for i in range(0, len(channels)):
            channels[i] = ~channels[i]
    composite = np.array(channels[0].shape, dtype=np.uint64)
    composite.fill(0)
    for channel in channels:
        intensities.append(channel_intensity(channel, bit_depth))
        composite = composite + channel
    intensities.append(channel_intensity(composite, bit_depth * len(intensities)))
    return intensities, len(channels[0])


def channel_intensity(channel, bit_depth=255):
    """
    Implementation of equation 3 for the normalized average pixel intensity, to be used in calculating image fractal
    dimension.
    [Reference]
    :param channel: ndarray
        Color channel to be averaged.
    :param bit_depth: int (default: 255)
        The maximum possible value of the pixels.
    :return: float
        Normalized average intensity for the color channel.
    """
    return (1 / bit_depth) * (np.sum(channel) / len(channel))


def channel_fds(intensities, epsilon, dimensions=None):
    """
    Calculates the fractal dimension of each channel and returns it in the order of the channels.
    :param intensities: list[floats]
        The normalized pixel intensities in each channel of the image.
    :param epsilon: int
        The number of boxes, i.e. the number of pixels in the image.
    :param dimensions: list[float]
        If supported dimensions are not adequate, specify appropriate dimensionality.
    :return: list[float]
        List of fractal dimensions for each channel.
    """
    fd = []
    supported_dims = {
        1: [2.0, 3.0],
        2: [2.0, 2.0, 4.0],
        3: [5.0, 5.0, 5.0, 5.0],
    }
    if dimensions is None:
        dimensions = supported_dims.get(len(intensities) - 1, None)
        if dimensions is None:
            raise ValueError("Image is unsupported, it has too many channels! Please try again!")
    for dimension, intensity in zip(dimensions, intensities):
        fd.append(channel_fd(dimension, intensity, epsilon))
    return fd


def channel_fd(dimension, intensity, epsilon):
    """
    Implementation of equation six and seven to calculate the fractal dimension of the image channels.
    [Reference]
    :param dimension: float
        The maximum possible dimension of the image, i.e. a color image is up to 5.0, and so on.
    :param intensity: float
        The normalized average pixel intensity for a single channel, or all channels.
    :param epsilon: int
        The number of boxes, i.e. the total number of pixels in the image.
    :return: float
        The estimated fractal dimension of the color image.
    """
    try:
        return -dimension * (math.log(epsilon * intensity)/math.log(1/epsilon))
    except ValueError:
        return np.NaN
    except ZeroDivisionError:
        return 0.0


def apv_dimension(fractal, image_flag=1, bit_depth=255, invert=False, alpha_channel=False, alpha_index=None, dimensions=None):
    """
    Implementation of a generalized fractal dimension estimation method for color, digital images, including alpha
    mask filtering.
    :param fractal: str or ndarray
        Full path to the image on disk or an array of values.
    :param image_flag: int or str
        The image reading mode, same as OpenCV Imread Modes, or 'arr' for
        Source: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
    :param bit_depth: int
        The maximum possible value of the pixels.
    :param invert: bool
        If true, the image is inverted when processing.
    :param alpha_channel: bool
        If true and alpha_index is None, the last channel of the image is used as alpha channel.
    :param alpha_index: int
        The index of the alpha channel, if None, then the last channel is used.
    :param dimensions: list
        List of expected maximum fractal dimensions ordered in the order of the fractal color channels
    :return: list[float]
        List of estimated fractal dimensions for each channel of the image.
    """
    assert image_flag == 'arr' or image_flag == 1 or image_flag == 0 or image_flag == -1, "Image format is not " \
                                                                                          "supported, try again."
    if image_flag != 'arr':
        img = cv2.imread(fractal, image_flag)
    else:
        img = fractal
    shape = img.shape
    if len(shape) > 2:
        img_pixels = np.split(img, img.shape[2], axis=2)
    else:
        img_pixels = img
    channels = flatten_image(img_pixels)
    if alpha_channel is True and alpha_index is None and len(shape) > 2:
        channels = alpha_filter(channels[0:-1], channels[-1])
    elif alpha_channel is True and alpha_index is not None and len(shape) > 2:
        channels = alpha_filter(np.append(channels[0:alpha_index], channels[alpha_index + 1:]), channels[alpha_index])
    intensities, epsilon = channel_intensities(channels, bit_depth, invert)
    return channel_fds(intensities, epsilon, dimensions)
