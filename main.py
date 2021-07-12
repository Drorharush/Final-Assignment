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
    size = np.shape(images)
    img_num = np.shape(images)[0]
    variance = np.array([np.sum(f * p ** 2 + B ** 2) for p in psfs])
    flipped_psfs = np.flip(psfs, axis=(1, 2))
    sum_term = [(convolve2d(images[i], flipped_psfs[i], mode='same') / variance[i]) for i in tqdm(range(img_num))]
    return np.sum(sum_term, axis=0)


def crop_center(image, cropx, cropy):
    y, x = np.shape(image)
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return image[starty:starty + cropy, startx:startx + cropx]


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
print(f'Test q=(50,50): {T}\n Test map at (50,50): {T_m[60, 60]}')
plt.imshow(T_m)
plt.show()
