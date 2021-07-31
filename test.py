import numpy as np
from tqdm import tqdm
from scipy.signal import convolve


def test_map(images, psfs, flux, B):
    """
    Returns the test statistic as a function of the planet position
    :param images: A list of same-size images
    :param psfs: A list of same-size PSFs for each image
    :param flux: The number of photons received
    :param B: The standard deviation for the white noise
    :return: A matrix with the same size as the images
    """
    psfs = np.array(psfs)
    images = np.array(images)
    flipped_psfs = np.flip(psfs, axis=(1, 2))

    subtrackted_images = images - flux * psfs
    sigma_squared = flux * psfs + B ** 2
    unnormalized_test_map = np.sum(conv_per_layer(subtrackted_images / sigma_squared, flipped_psfs)
                                   - psfs * subtrackted_images / sigma_squared, axis=0)

    # Calculating the variance:
    term1 = conv_per_layer(flipped_psfs ** 2, 1 / sigma_squared)
    term2 = conv_per_layer(flipped_psfs, psfs / sigma_squared)
    term3 = psfs ** 2 / sigma_squared
    variance = np.sum(term1 - 2 * term2 + term3, axis=0)
    return (variance ** - 0.5) * unnormalized_test_map


def conv_per_layer(matrix1, matrix2):
    """
    Convolves two matrices per layer (axis=0) and returns the result.
    :param matrix1: 3dim matrix
    :param matrix2: 3dim matrix of the same shape as matrix1
    :return: matrix of the same transverse shape as matrix1
    """
    assert np.shape(matrix1)[0] == np.shape(matrix2)[0], 'The matrices have different number of layers'

    layers = np.shape(matrix1)[0]
    return np.array([convolve(matrix1[i], matrix2[i], mode='same') for i in tqdm(range(layers), position=0, leave=True)])
