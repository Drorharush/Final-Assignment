import random
from scipy.stats import percentileofscore, norm
from scipy.special import erfinv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from test import conv_per_layer, test_map
from posterior_estimation import tiny_shift, posterior
from tqdm import tqdm
import corner

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


def plot_image(image, title, mark=False, coordinates=np.array([0, 0])):
    plt.imshow(image)
    plt.colorbar()
    plt.title(title)
    if mark:
        plt.scatter(coordinates[0], coordinates[1], edgecolors='red', s=40, facecolors='none',
                    label=f'Planet coordinates ({coordinates[0]}, {coordinates[1]})')
        plt.legend()
    plt.axis('off')
    plt.savefig(title + '.jpg')
    plt.show()


def h0_factory(size, number, psfs, flux, noise_sigma):
    """
    Creating a number of sky images with no planet
    :param size: Two integers for the size of the images
    :param number: The number of images to create, int.
    :param psfs: The PSFs to use
    :param flux: A number representing the flux
    :param noise_sigma: A number representing the noise STD
    :return: A tuple, the PSFs used for each image and the images.
    """
    psf_array = np.array(psfs)
    rows, columns = size
    rng = np.random.default_rng()
    white_noise = noise_sigma * rng.standard_normal(size=(number, rows, columns))
    if number > np.shape(psf_array)[0]:
        # Creating the PSF array used to create H0 images from the original PSFs.
        rnd_index = np.random.randint(np.shape(psf_array)[0], size=number)
        new_psfs = np.array([psf_array[rnd_index[i]] for i in range(number)])
        shot_noise = np.sqrt(flux * new_psfs) * rng.standard_normal(size=np.shape(new_psfs)) + f * new_psfs
        return new_psfs, shot_noise + white_noise
    shot_noise = np.sqrt(flux * psf_array)[:number] * rng.standard_normal(size=np.shape(psf_array[:number])) \
                 + f * psf_array[:number]
    return psfs[:number], shot_noise + white_noise


def plot_hist(title, H0_scores, H1_scores, beta1=0, log=False, plot_PDF=False, mu=0, sigma=1):
    fig = plt.figure(tight_layout=True)
    plt.title(title)
    bins = min(25, max(len(H0_scores) // 4, 3))
    if log:
        H0_scores, H1_scores = np.log(H0_scores), np.log(H1_scores)
    if beta1 > 0:
        eta = np.quantile(H0_scores, 1 - beta1)
        beta0 = percentileofscore(H1_scores, eta) / 100
        eta_label = rf'$ln(\eta) \approx {round(eta)}$' if log else rf'$\eta = {eta}$'
        eta_label += f'\n' + rf'$\beta_0 = {beta0}$'
        plt.axvline(x=eta, label=eta_label, color='k', ls='--')

    plt.hist(H0_scores, bins=bins, density=True, alpha=0.5, color='r', label=r'$H_0$')
    if H0_scores.any() != H1_scores.any():
        plt.hist(H1_scores, bins=bins, density=True, alpha=0.5, color='b', label=r'$H_1$')
    if plot_PDF:
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        plt.plot(x, p, color='k', linewidth=2, label=f'T|H0 ~ N({mu}, {round(sigma ** 2, 7)})')
        fitted_mu, fitted_std = norm.fit(H1_scores[np.isfinite(H1_scores)])
        fitted_gaussian = norm.pdf(x, fitted_mu, fitted_std)
        plt.plot(x, fitted_gaussian, color='g', alpha=0.5,
                 label=f'Fitted N({round(fitted_mu, 2)}, {round(fitted_std, 2)})')
    x_label = 'Log test scores' if log else 'Test scores'
    plt.xlabel(x_label)
    plt.ylabel('Probability')
    plt.legend()
    filename = title + '.jpg'
    plt.savefig(filename)
    plt.show()


def threshold(false_positive):
    return norm.isf(false_positive)


def H1_variance(psfs, flux, STD):
    psfs = np.array(psfs)
    sigma_squared = flux * psfs + STD ** 2
    flipped_psfs = np.flip(psfs, axis=(1, 2))
    term1 = conv_per_layer(flipped_psfs ** 2, 1 / sigma_squared)
    term2 = conv_per_layer(flipped_psfs, psfs / sigma_squared)
    term3 = psfs ** 2 / sigma_squared
    return np.sum(term1 - 2 * term2 + term3, axis=0)


def minimal_flux(psfs, flux, STD, eta, variance_map):
    epsilon_map = (H1_variance(psfs, flux, STD) ** -0.5) * eta / flux
    center_row, center_col = np.shape(epsilon_map)[0] // 2, np.shape(epsilon_map)[1] // 2
    r_lim = 100
    return epsilon_map[center_row:center_row + r_lim, center_col]


def plot_posterior(post_matrix, qi_values, qj_values, epsilon_values):
    qi_post = np.sum(post_matrix, axis=(1, 2))
    qj_post = np.sum(post_matrix, axis=(0, 2))
    eps_post = np.sum(post_matrix, axis=(0, 1))
    qi_qj = np.sum(post_matrix, axis=2)
    qi_eps = np.sum(post_matrix, axis=1)
    qj_eps = np.sum(post_matrix, axis=0)
    fig_post, axs = plt.subplots(2, 3, figsize=(9, 6))
    fig_post.suptitle('Posterior')

    axs[0, 0].plot(qi_values, qi_post)
    axs[0, 0].set_title(r'$P(q_i)$')
    max_i_pixel = qi_values[np.argmax(qi_post)]
    axs[0, 0].axvline(x=max_i_pixel, label=f'max = {round(max_i_pixel, 2)}', color='k', ls='--')
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].legend()

    axs[0, 1].plot(qj_values, qj_post)
    axs[0, 1].set_title(r'$P(q_j)$')
    max_j_pixel = qj_values[np.argmax(qj_post)]
    axs[0, 1].axvline(x=max_j_pixel, label=f'max = {round(max_j_pixel, 2)}', color='k', ls='--')
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].legend()

    axs[0, 2].plot(epsilon_values, eps_post)
    axs[0, 2].set_title(r'$P(\epsilon)$')
    axs[0, 2].ticklabel_format(axis='x', style='sci')
    max_eps = epsilon_values[np.argmax(eps_post)]
    axs[0, 2].axvline(x=max_eps, label=f'max = {round(max_eps  * 10 ** 4, 2)}E-4', color='k', ls='--')
    axs[0, 2].set_ylim(bottom=0)
    axs[0, 2].legend()

    axs[1, 0].imshow(qi_qj, extent=(qj_values[0], qj_values[-1], qi_values[0], qi_values[-1]), aspect='auto')
    axs[1, 0].set_title(r'$q_i$ vs $q_j$')

    axs[1, 1].imshow(qi_eps, extent=(epsilon_values[0], epsilon_values[-1], qi_values[0], qi_values[-1]), aspect='auto')
    axs[1, 1].set_title(r'$q_i$ vs $\epsilon$')

    axs[1, 2].imshow(qj_eps, extent=(epsilon_values[0], epsilon_values[-1], qj_values[0], qj_values[-1]), aspect='auto')
    axs[1, 2].set_title(r'$q_j$ vs $\epsilon$')

    plt.tight_layout()
    plt.savefig('Posterior.jpg')
    plt.show()
    pass


img, psf = folder_to_list(img_folder), folder_to_list(psf_folder)
img_size, psf_size = np.shape(img[0]), np.shape(psf[0])
img_center, psf_center = [img_size[0] // 2, img_size[1] // 2], [psf_size[0] // 2, psf_size[1] // 2]

f = 10 ** 7
f_array = np.array([10 ** 7, 10 ** 8, 10 ** 9])
B = 1
beta1 = 1 / (200 * np.ceil(np.pi * (100 ** 2)))
eta = norm.isf(beta1)
# plt.imshow(img[0])
# plt.show()

# checking the flux
# f_approx = np.sum(img) / np.shape(img)[0]
# print(f'flux = {f_approx}')

# Data test map
# data_test_map = test_map(img, psf, f, B)
# map_no_nan = data_test_map.copy()
# map_no_nan[np.isnan(map_no_nan)] = 0
# q = np.unravel_index(np.argmax(map_no_nan, axis=None), data_test_map.shape)
# maximal_test = data_test_map[q]
# print(f'Data test = {maximal_test}')
# plot_image(data_test_map, f'Data test map', mark=True, coordinates=q)


# H0 histogram

# h0_psf, h0_img = h0_factory(img_size, 40, psf, f, B)
# h0_map = test_map(h0_img, h0_psf, f, B)
# plot_image(h0_map, f'H0 test map shape{np.shape(h0_map)}')
# plot_hist('H0 Scores', h0_map.flatten(), h0_map.flatten(), plot_PDF=True, mu=0, sigma=1)


# testng H1_variance
# var = H1_variance(psf, f, B)
# plt.imshow(var)
# plt.show()

# epsilin min vs |q|
# e_vs_q = np.zeros([3, 100])
# for i in range(f_array.shape[0]):
#     var = H1_variance(psf, f_array[i], B)
#     e_vs_q[i] = minimal_flux(psf, f_array[i], B, eta, var)
#     plt.plot(e_vs_q[i], label=f'f = 10 ** {np.log10(f_array[i])}', lw=2)
#     plt.yscale('log')
# plt.title(r'$\varepsilon_{min}$ vs |$\vec{q}$|', fontsize=18)
# plt.xlim((0, 100))
# plt.ylim(bottom=0)
# plt.ylabel(r'$\varepsilon_{min}$')
# plt.xlabel(r'|$\vec{q}$|')
# plt.legend()
# plt.tight_layout()
# plt.savefig('eps_vs_q')
# plt.show()

# initial guess eps
# var = H1_variance(psf, f, B)
# e_vs_q = minimal_flux(psf, f, B, eta, var)
# print(e_vs_q[15])

# Testing tiny_shift - WORKS!
# sample_img = img[5]
# shift = (8, 13)
# shifted_sample = tiny_shift(sample_img, shift)
# plot_image(sample_img, 'sample img')
# plot_image(shifted_sample, 'shifted sample')

# testing posterior:
samples_per_axis = 20
q_i_values = np.linspace(7, 9, samples_per_axis)
q_j_values = np.linspace(6, 7, samples_per_axis)
eps_values = np.linspace(2 * 10**-4, 3.5*10**-4, samples_per_axis)
post, log_post = posterior(img, psf, f, B, q_i_values, q_j_values, eps_values)

plot_posterior(post, q_i_values, q_j_values, eps_values)

# plot_posterior(log_post, q_i_values, q_j_values, eps_values)

