import numpy as np
from tqdm import tqdm


def tiny_shift(matrix, shift):
    """
    shifts a matrix by a given shift, that can be smaller than one pixel.
    :param matrix: 2D matrix
    :param shift: (row_shift, column_shift)
    :return: 2D matrix of the same size as 'matrix'
    """
    fft_matrix = np.fft.rfft2(matrix)
    f_i = np.fft.fftfreq(fft_matrix.shape[0])
    f_j = np.fft.fftfreq(fft_matrix.shape[1])
    fi_g, fj_g = np.meshgrid(f_i, f_j, sparse=False, indexing='ij')
    shifted_matrix = np.fft.irfft2(np.exp(-2*np.pi*1j*(shift[0]*fi_g + shift[1]*fj_g)) * fft_matrix, matrix.shape)
    return shifted_matrix


def log_posterior(images, psfs, flux, sigma, q, eps):
    images = np.array(images)
    psfs = np.array(psfs)
    shifted_psfs = np.array([tiny_shift(psf, q) for psf in psfs])
    subtracted_images = images - (flux * (1-eps) * psfs + flux * eps * shifted_psfs)
    ln_p = np.sum(np.log(1 / (sigma * 2 * np.pi)) - 0.5 * (subtracted_images / sigma) ** 2)
    return ln_p


def posterior(images, psfs, flux, B, q_i_array, q_j_array, eps_array):
    """
    Returns P(qx, qy, eps | data) as a matrix
    :param images: matrix shaped (img_num, i, j)
    :param psfs: matrix shaped (psf_num, i, j)
    :param flux: A positive number
    :param B: A positive number
    :param q_i_array: 1D array
    :param q_j_array: 1D array
    :param eps_array: 1D array
    :return: 3D matrix shaped (qx, qy, eps)
    """
    log_p = np.zeros([len(q_i_array), len(q_j_array), len(eps_array)])
    psfs = np.array(psfs)
    images = np.array(images)
    sigma = np.sqrt(flux * psfs + B ** 2)
    for i, q_i in (enumerate(tqdm(q_i_array))):
        for j, q_j in enumerate(q_j_array):
            for k, eps in enumerate(eps_array):
                q = np.array([q_i, q_j])
                log_p[i, j, k] = log_posterior(images, psfs, flux, sigma, q, eps)
    log_p = log_p - np.max(log_p)
    p = np.exp(log_p) / np.sum(np.exp(log_p))
    return p, log_p
