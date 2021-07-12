import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve2d

psf_folder = Path('.\HCI_data\psfs')
img_folder = Path('.\HCI_data\images')


def folder_to_list(folder: Path) -> list:
    """
    Takes a Path object of a folder and returns a list of the .npy files
    :param folder: Directory Path object
    :return: A list
    """
    assert folder.is_dir(), 'The path given is not a folder'
    return [np.load(file) for file in sorted(folder.glob('*.npy'))]


def Test(images, psfs, q, f, B, precalc_var=np.array(-1)):
    if np.all(precalc_var == -1):
        variance = np.array([np.sum(f * p ** 2 + B ** 2) for p in psfs])
    else:
        variance = precalc_var
    mu_I = np.array([np.sum(np.roll(psfs, q, axis=(0, 1)) * image) for image in images])
    return np.sum(mu_I / variance)


def Test_map(images, psfs, f, B):
    img_num = np.shape(images)[0]
    variance = np.array([np.sum(f * p ** 2 + B ** 2) for p in psfs])
    flipped_psfs = np.flip(psfs, axis=(1, 2))
    sum_term = [fast_conv(images[i], flipped_psfs[i]) / variance[i] for i in tqdm(range(img_num))]
    return np.sum(sum_term, axis=0)


def fast_conv(a, b):
    return np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(a) * np.fft.rfft2(b)))


def crop_center(image, cropx, cropy):
    y, x = np.shape(image)
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return image[starty:starty + cropy, startx:startx + cropx]


def plot_image(image, title):
    plt.imshow(image)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(title + '.jpg')
    plt.show()


f = 10 ** 7
B = 1

img, psf = folder_to_list(img_folder), folder_to_list(psf_folder)
img_size, psf_size = np.shape(img[0]), np.shape(psf[0])
img_center, psf_center = [img_size[0] // 2, img_size[1] // 2], [psf_size[0] // 2, psf_size[1] // 2]

# Testing flip
# flipped_psfs = np.flip(psf, axis=(1, 2))
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('flipped and non flipped PSF')
# ax1.imshow(psf[9])
# ax1.set_title('PSF')
# ax2.imshow(flipped_psfs[9])
# ax2.set_title('flipped PSF')
# plt.show()

# Testing Test_map
T = Test(img, psf, [img_center[0] + 10, img_center[1] + 10], f, B)
T_m = Test_map(img, psf, f, B)
print(f'Test q=(10,10): {T}\n Test map at q=(10,10): {T_m[img_center[0] + 10, img_center[1] + 10]}')
plot_image(T_m, 'Test map')

# Testing conv2d vs fft
# variance = np.array(np.sum(f * psf[0] ** 2 + B ** 2))
# flipped_psf = np.flip(psf[0])
# conv_fft = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(img[0]) * np.fft.rfft2(flipped_psf)))
# map_fft = conv_fft / variance
# conv_normal = convolve2d(img[0], flipped_psf, mode='same')
# map_conv = conv_normal / variance
#
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('fft and conv2d')
# ax1.imshow(map_fft)
# ax1.set_title('fft')
# ax2.imshow(map_conv)
# ax2.set_title('conv2d')
# plt.show()
#
