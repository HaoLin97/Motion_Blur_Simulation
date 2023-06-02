import cv2
import numpy as np


def crop_roi(input_image):
    # dim = (int(width / 4), int(height / 4))
    # image = cv2.resize(input_image, dim)
    # roi = cv2.selectROI(image)
    # print(roi)
    # y1 = roi[0]
    # y2 = roi[0]+roi[2]
    # x1 = roi[1]
    # x2 = roi[1]+roi[3]
    # example y1,y2,x1,x2 value = (360,420,440,500)
    y1, y2, x1, x2 = (360 * 4, 420 * 4, 440 * 4, 500 * 4)

    image = input_image[x1:x2, y1:y2]
    cv2.imshow("0", image)
    cv2.waitKey()
    return image
    pass

def snr_calc(image):

    sd = np.std(image)
    mean = np.mean(image)

    snr = mean/sd
    print(snr)
    return snr



if __name__ == '__main__':

    # image_path = "slanted_edge_1x2_cratio_4_gamma_1.png"
    image_path = "slanted_exp_30ms_dx_0_dy_0_lux_50_PE_5774.png"
    blurred_path = "slanted_exp_30ms_dx_990_dy_0_lux_50_PE_5774.png"
    # blurred_path = "slanted_exp_5ms_dx_0_dy_0_lux_50_PE_962.png"
    image = cv2.imread(image_path, -1)
    print(image.shape)
    cropped = crop_roi(image)

    ideal_snr = snr_calc(cropped)

    blurred_image = cv2.imread(blurred_path, -1)
    blurred_cropped = crop_roi(blurred_image)
    blur_snr = snr_calc(blurred_cropped)

    print("Difference in snr is {}".format(ideal_snr-blur_snr))

    pass
