import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import plotly.graph_objects as go

from natsort import natsorted as ns
import os

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
    # cv2.imshow("0", image)
    # cv2.waitKey()
    return image
    pass

def snr_calc(image):

    sd = np.std(image)
    mean = np.mean(image)

    snr = mean/sd
    # print(snr)
    return snr



if __name__ == '__main__':

    # image_path = "slanted_edge_1x2_cratio_4_gamma_1.png"
    # image_path = "slanted_exp_30ms_dx_0_dy_0_lux_50_PE_5774.png"
    # blurred_path = "slanted_exp_30ms_dx_990_dy_0_lux_50_PE_5774.png"
    # # blurred_path = "slanted_exp_5ms_dx_0_dy_0_lux_50_PE_962.png"
    # image = cv2.imread(image_path, -1)
    # print(image.shape)
    # cropped = crop_roi(image)
    #
    # ideal_snr = snr_calc(cropped)
    #
    # blurred_image = cv2.imread(blurred_path, -1)
    # blurred_cropped = crop_roi(blurred_image)
    # blur_snr = snr_calc(blurred_cropped)
    #
    # print("Difference in snr is {}".format(ideal_snr-blur_snr))
    path = "F:/Pycharm_Sim_Results/new_model/old_slanted/slanted/50lux_30ms"
    dst = "F:/Pycharm_Sim_Results/new_model/new_stop/"

    images = []
    for f in os.listdir(path):
        if f.endswith("png"):
            # if ("_50lux" in f) or ("_100lux" in f) or ("_20lux" in f):
            images.append(f)
    snrs = []
    EXP = []
    for image in images:
        # exp = image.split(sep="_")[2].replace("ms", "")
        exp = image.split(sep="_")[4]
        # print(dx)
        image_path = os.path.join(path, image)
        cropped = crop_roi(cv2.imread(image_path, -1))
        snr = snr_calc(cropped)
        snrs.append(snr)
        EXP.append(exp)


    # print(snrs)

    df = DataFrame({'EXP': EXP,
                    'snr': snrs})
    df['EXP'] = pd.to_numeric(df['EXP'])

    df.sort_values('EXP', ascending=True, inplace=True)

    df.to_excel("excel_30ms_50lux_patchsnr.xlsx", sheet_name="sheet1")
    plt.rcParams['figure.figsize'] = [12, 9]
    data = df

    # # Plot
    # plt.rcParams['figure.figsize'] = [12, 9]
    # plt.rcParams.update({'font.size': 22})
    # plt.ylim([0, 16])
    # plt.xlim([0, 1000])
    # plt.plot(data.EXP, data.snr)
    # # plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    # plt.title("Performance Graph of DX vs PatchSNR for 30ms 50lux", y=1.05)
    # plt.xlabel("Horizontal Pixel Per Sec (DX)")
    # plt.ylabel("Patch SNR")
    # plt.savefig("DX vs PatchSNR.png")
    # plt.close()

    px_sizes = [32, 50, 100]

    for px_size in px_sizes:
        for blur in [0]:
            det_path = "F:/Pycharm_Sim_Results/new_model/new_stop/archive/detect/{0}x{0}_50lux_30ms/labels".format(
                px_size)

            results = []
            metrics = ["Confidence"]
            for f in os.listdir(det_path):
                if f.endswith(".txt"):
                    results.append(f)

            results = ns(results)

            plt.rcParams['figure.figsize'] = [15, 12]
            plt.rcParams.update({'font.size': 22})
            # plt.ylim([0, 1])
            # plt.xlim([0, 1000])
            exposure = []
            Confidence = []
            for file in results:

                with open(os.path.join(det_path, file)) as f:
                    det_data = f.readline()
                    if int(det_data.split(' ')[0]) == 11:
                        Confidence.append(float(det_data.split(' ')[-1].replace('\n', '')))
                    else:
                        Confidence.append(0)
                    exposure.append(int(file.split('_')[3].replace("ms", "")))
                    # exposure.append(int(file.split('_')[5]))

            Confidence = Confidence[:90]

            fig = go.Figure(data=
                            go.Contour(x=data.EXP[:len(Confidence)], y=data.snr[:len(Confidence)], z=Confidence,
                                       contours=dict(start=0, end=1, size=0.05,
                                                     showlabels=True,  # show labels on contours
                                                     labelfont=dict(  # label font properties
                                                         size=12,
                                                         color='white',
                                                     )),
                                       colorbar=dict(
                                           title="Confidence",
                                           titleside='right',
                                           titlefont=dict(
                                               size=14,
                                               family='Arial, sans-serif'))))

            fig.update_layout(
                title="System Performance Map of {} vs Motion Blur vs Confidence <br> for {} px @30ms".format("Patch SNR".upper(), px_size),
                xaxis_title="Horizontal Pixels Per Second (DX)",
                yaxis_title="{}".format("Patch SNR"))
            # # fig.show()
            fig.write_image("{}/Contour_Graph_blur_{}_{}.png".format(dst, "PATCH_SNR".upper(), px_size), engine="kaleido")


    pass
