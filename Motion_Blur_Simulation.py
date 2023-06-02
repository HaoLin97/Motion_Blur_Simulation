import numpy as np
import cv2
import os
import scipy
import skimage.morphology
import skimage.filters
import math
from math import pi as pi
from scipy.integrate import quad, simps
from scipy.constants import h, c, k
import pandas as pd

PE = 0

def check_clipping(to_check, limit=65535):
    row, col = to_check.shape
    checked = to_check
    to_check[to_check > limit] = limit
    to_check[to_check < 0] = 1

    return checked


def integral_approximation(f, a, b):
    return (b - a) * np.mean(f)


def matrix_multiply(a, b):
    return np.multiply(np.asarray(a), np.asarray(b))


def calc_Ap(x, y, r, f):
    """
    function to calculate Ap, the apparant area of the pixel project into object space
    inputs:
    x, y: pixel dimensions (typically in um)
    r: object distance
    f: focal length

    Returns:
    Ap

    """
    return (x * y * r ** 2) / f ** 2


def calc_Omega_L(d, r):
    """
    function to calculate the solid angle of the lens
    inputs:
    d: lens diameter
    r: object distance

    returns:
    Omega_L: the solid angle in radians
    """
    return ((pi * d ** 2) / 4 * r ** 2)


def calc_Lp(Ls, x, y, to, ta, f):
    """
    calculates the illuminance in a pixel, for a given scene luminance
    inputs:
    Ls: source luminance (cd/m^2)
    x,y: pixel dimensions
    to: optical attenuation
    ta: atmospheric attenuation
    f: lens f#
    """
    return (Ls * x * y * pi * to * ta) / (4 * f ** 2)


def planck(wav, T):
    a = 2.0 * h * c ** 2
    b = h * c / (wav * k * T)
    intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))
    return intensity

# Blackbody curve as calculated by Richardson, 2002


def richardson(wav, T):
    C1 = 3.74e8
    C2 = 1.44e4

    a = C1
    b = C2 / (wav * T)
    intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))

    return intensity


def lux_to_quanta(lens_f=1.4, lens_focal_length=15e-3, pixel_pitch=3.45e-6, to=1.0, ta=1.0, T_int=0.03, E_amb=200.0,
                  lambda_min=400, lambda_max=650, Kamb=5500, K_m=683):
    """
    Converts the ambient lux into quanta that hits the sensor
    :param lens_f: F number of the lens
    :param lens_focal_length: Focal Length of the lens
    :param pixel_pitch: Size of pixel
    :param to: Transmission of the optics
    :param ta: Transmission of the atmosphere
    :param T_int: Exposure time
    :param E_amb: Ambient Lux
    :param lambda_min: Min of the wavelength
    :param lambda_max: Max of the wavelength
    :param Kamb: Ambient light temperature
    :param K_m: define Km, the maximum luminous efficacy (683lmW^01 @ 555nm for photopic, 1700lmW^01 @ 510nm for scotopic)
    :return: The quanta at the sensor given the camera setting, exposure and ambient lighting
    """

    # lens_d = lens_focal_length / lens_f

    # read in the V(λ) csv file. Default option is photopic
    V_lambda = pd.read_csv('spectra/luminous efficacy/Photopic Luminous Efficacy.csv')
    V_lambda.columns = ['wavelength', 'value']
    V_range = np.where(np.logical_and(V_lambda.wavelength >= lambda_min, V_lambda.wavelength < lambda_max))
    V_lambda = np.asarray(V_lambda.value)[V_range]

    # read in a spectrum. For this example, we'll choose D55
    D55 = pd.read_csv('spectra/illuminant/interpolated/D55.csv')
    D55.columns = ['wavelength', 'value']
    D55_range = np.where(np.logical_and(D55.wavelength >= lambda_min, D55.wavelength < lambda_max))

    # define the illuminant spectral power in Watts
    # a = np.asarray(D55.value)[D55_range]
    # b = V_lambda
    # Phi_V = K_m * simps((a * b), x=range(lambda_min, lambda_max), even="avg")

    W_wavelength = np.arange(lambda_min * 1e-9, (lambda_max) * 1e-9, 1e-9)
    W_lambda = planck(W_wavelength, Kamb)

    # scale so that the power within the wavelength range is 1 Watt

    W_power = simps(W_lambda, x=range(lambda_min, lambda_max), even="avg")

    W_lambda = W_lambda / W_power

    # confirm the area under the curve is now 1
    # print(simps(W_lambda, x=range(lambda_min, lambda_max), even="avg"))

    # Commented out because the reflection is taken care of in the simulated image
    # read in the chart reflection, S. In this example, we'll use patch 22
    # S = pd.read_csv('spectra/Xrite/interpolated/22_neutral_070D.csv')
    # S.columns = ['wavelength', 'value']
    # S_range = np.where(np.logical_and(S.wavelength >= lambda_min, S.wavelength < lambda_max))
    # S_lambda = np.asarray(S.value)[S_range]

    # read in the QE curve. In this example, we'll use the mono cfa
    Q = pd.read_csv('spectra/QE/ar0237_qe_curves.csv')
    # Q = pd.read_csv('spectra/QE/interpolated/QE_green.csv')
    Q.columns = ['wavelength', 'red', 'green', 'blue']
    Q_range = np.where(np.logical_and(Q.wavelength >= lambda_min, Q.wavelength < lambda_max))
    Q_lambda = np.asarray(Q.green)[Q_range]

    # read in the IR cut-off filter
    I = pd.read_csv('spectra/QE/interpolated/QE_IRCF.csv')
    I.columns = ['wavelength', 'value']
    I_range = np.where(np.logical_and(I.wavelength >= lambda_min, I.wavelength < lambda_max))
    I_lambda = np.asarray(I.value)[I_range]

    # Calculate E_source
    E_source = K_m * simps(matrix_multiply(W_lambda, V_lambda), x=range(lambda_min, lambda_max), even="avg")
    # calculate the scale value to convert from lux back to Watts
    E_scale = E_amb / E_source

    # calculate P_lambda, the spectral irradiance available to the sensor
    # horrible daisychain of matrix multiplications...
    P_lambda = E_scale * matrix_multiply(W_lambda, matrix_multiply(I_lambda, Q_lambda))
    P_wavelength = W_wavelength

    Pp_lambda = (P_lambda * pixel_pitch * pixel_pitch * ta * to) / (4 * (lens_f ** 2))

    # calculate epsilon_lambda
    e_lambda = np.divide(h * c, P_wavelength)

    # calculate PEp, the number of photons captured by the pixel
    PEp = T_int * simps(np.divide(Pp_lambda, e_lambda), x=range(lambda_min, lambda_max), even="avg")
    global PE
    PE = int(PEp)
    # print(PEp)
    print("Lux:{}, Tint:{}, PEp:{}".format(lux, T_int, PEp))
    return PEp


def add_camera_noise(input_irrad_photons, qe=1, sensitivity=5.88, dark_noise=2.44, bitdepth=12, baseline=0,
                     saturation_capacity=11038, rs=np.random.RandomState(seed=42)):
    """
    Function to add camera noise to input image, models the camera sensor
    :param input_irrad_photons: The input image
    :param qe: taken care of in lux conversion in motion blur method so now set to 1
    :param sensitivity: The sensitivity of the camera represents the amplification of the voltage in the pixel from the
    photoelectrons and is also a property of the camera
    :param dark_noise: The dark noise is comprised of two separate noise sources, the read noise and the dark current
    :param bitdepth: bit depth of the sensor
    :param baseline: the fixed offset within the camera
    :param saturation_capacity: saturation capacity describes the number of electrons that an individual pixel can store
    :param rs: random seed
    :return: Image with added camera noise
    """
    input_irrad_photons = input_irrad_photons / (qe * sensitivity)

    # Add shot noise
    photons = rs.poisson(input_irrad_photons, size=input_irrad_photons.shape)

    # Convert to electrons
    electrons = qe * photons

    # Add dark noise
    dark_noise = rs.normal(scale=dark_noise, size=electrons.shape)
    electrons_out = dark_noise + electrons
    # cv2.imwrite("added_noise.png", electrons_out.astype(np.uint16))
    electrons_out = check_clipping(electrons_out, saturation_capacity)

    # Convert to ADU and add baseline
    max_adu = np.int(2**bitdepth - 1)
    adu = (electrons_out * sensitivity).astype(np.uint16) # Convert to discrete numbers
    adu += baseline
    adu[adu > max_adu] = max_adu # models pixel saturation
    adu[adu < 0] = 0 # models pixel saturation

    # Bit shift the image by 4 bits to make it 16 bit image
    return np.left_shift(adu.astype(np.uint16), 4)


def add_motion_blur(input_image, exposure_time=0.03, dx_dt=0, dy_dt=0, lux=200):
    """
    Add motion blur to the input image
    :param input_image: Input Image
    :param exposure_time: Exposure time to simulate
    :param dx_dt: Pixel movement per second in horizontal axis
    :param dy_dt: Pixel movement per second in vertical axis
    :param lux: Ambient Lux
    :return: Motion blurred image
    """
    # cv2.imwrite("input.png", input_image)
    if len(input_image.shape) == 3:
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        gray[gray == 0] = 1
        target_rows, target_cols = gray.shape

        # cv2.imwrite("gray.png", gray)
    else:
        target_rows, target_cols = input_image.shape
        gray = input_image
    # Mask
    scale_factor = 2
    # Create 18% grey in mask
    mask = np.ones([round(target_rows*scale_factor, 0),
                   round(target_cols*scale_factor, 0)])*0.18*255

    [mask_rows, mask_cols] = mask.shape

    target_start_row = int(round((mask_rows / 2.0) - (target_rows / 2.0), 0))
    target_start_col = int(round((mask_cols / 2.0) - (target_cols / 2.0), 0))

    for i in range(target_start_row, target_start_row+target_rows-1):
        for j in range(target_start_col, target_start_col + target_cols - 1):
            mask[i, j] = gray[i-target_start_row, j-target_start_col]

    # Old Lux conversion - now redundant
    # photon_per_lux = 11300
    # dq_dt = photon_per_lux * lux
    quanta = lux_to_quanta(E_amb=lux, T_int=exposure_time)

    # Normalise the image and multiply it by quanta
    mask = np.round_((mask/(2**8)) * quanta, 0)

    # mask = check_clipping(mask, limit=255)

    # Multiplying by 256 to scale from 8 bit to 16 bit
    # image = mask.astype(np.uint16)*256


    image = mask.astype(np.uint16)
    # cv2.imwrite("masked.png", image)

    # Optical blur
    # Got psf array from matlab
    psf = np.array([[0.0250785810238330, 0.145343947430219, 0.0250785810238330],
            [0.145343947430219, 0.318309886183791, 0.145343947430219],
            [0.0250785810238330, 0.145343947430219, 0.0250785810238330]])
    # image_pre_optical = image
    # cv2.imwrite("optical_blur.png", image)
    image = scipy.ndimage.convolve(image, psf, mode='nearest')

    # Motion Blur

    # Calculating vertical and horizontal motion blur
    horizontal_motion = dx_dt*exposure_time
    vertical_motion = dy_dt*exposure_time

    # If there is any vertical or horizontal blur required
    motion_length = round(math.sqrt(vertical_motion * vertical_motion +
                                    horizontal_motion * horizontal_motion), 0)

    # if vertical_motion != 0 or horizontal_motion != 0:
    if motion_length > 1:

        if horizontal_motion == 0:
            motion_angle = 90
        else:
            motion_angle = np.rad2deg(math.atan(vertical_motion/horizontal_motion))

        # Initialising the blur kernel of the size of blur
        blur_kernal = np.zeros((int(motion_length), int(motion_length)), dtype=np.float32)

        # Creating kernel of ones at the center of the kernel
        blur_kernal[(int(motion_length) - 1) // 2, :] = np.ones(int(motion_length), dtype=np.float32)

        # Rotating the kernel to the direction/angle of blur
        # getRotationMatrix2D creates a transformation matrix, in which the kernel will be rotated by
        blur_kernal = cv2.warpAffine(blur_kernal,
                                     cv2.getRotationMatrix2D((int(motion_length) / 2, int(motion_length) / 2),
                                                             int(motion_angle), 1.0),
                                     (int(motion_length), int(motion_length)), flags=cv2.INTER_LINEAR)

        # Normalising the kernel
        blur_kernal = blur_kernal * (1.0 / np.sum(blur_kernal))
        # Applying blur kernel to the image
        image = cv2.filter2D(image, -1, blur_kernal)

    # image = check_clipping(image)

    # cv2.imwrite("motion_blur.png", image)
    return image


if __name__ == '__main__':
    # input = cv2.imread("exposure_30000_delay_0us_5.tiff", -1)
    # # input = cv2.imread("star_1600.tif", -1)
    # # input = cv2.imread("siemens_star_72.tif", -1)
    # # input = cv2.imread("ac6e638d-7c84846d.jpg", -1)
    #
    # blurred = add_motion_blur(input_image=input, dx_dt=0, dy_dt=0, lux=200, exposure_time=0.01)
    # # cv2.imwrite("blurred.png", blurred)
    #
    # output = add_camera_noise(blurred)
    # # filename = "star_exp_{}ms_dx_{}_dy_{}_lux_{}.png".format(int(e * 1000), x, y, lux)
    # cv2.imwrite("30ms_200lux_real_simto10ms.png", output)


    input = cv2.imread("slanted_edge_1x2_cratio_4_gamma_1.png", -1)
    # input = cv2.imread("star_1600.tif", -1)
    # input = cv2.imread("siemens_star_72.tif", -1)
    # input = cv2.imread("ac6e638d-7c84846d.jpg", -1)
    # exp = [0.01, 0.02, 0.03, 0.04]
    # luxs = [1, 5, 10, 20, 30, 40, 50, 100, 200]
    # blurs = [150, 500, 1000]
    # for blur in blurs:
    for blur in range(0, 1000, 10):
        # for lux in luxs:
            lux = 20
            y = 0
            x = blur
            exp = 30
            # exp_list = [10, 20, 30, 40]
            # dir = "H:/Pycharm_Sim_Results/new_model/slanted/{}lux_{}blur".format(lux, x)
            dir = "H:/Pycharm_Sim_Results/new_model/slanted/{}lux_{}ms".format(lux, exp)
            # for exp in range(0,100,1):
            if not os.path.exists(dir):
                os.makedirs(dir)
            e = exp/1000
            print(e)
            # for x in range(0, 1000, 10):

            blurred = add_motion_blur(input_image=input, dx_dt=x, dy_dt=0, lux=lux, exposure_time=e)
            # cv2.imwrite("blurred.png", blurred)

            output = add_camera_noise(blurred)
            filename = "slanted_exp_{}ms_dx_{}_dy_{}_lux_{}_PE_{}.png".format(int(e * 1000), x, y, lux, PE)
            print(filename)
            cv2.imwrite(os.path.join(dir, filename), output)

