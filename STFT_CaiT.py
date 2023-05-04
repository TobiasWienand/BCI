import random

from sklearn.model_selection import train_test_split
import numpy as np
from misc.Visualization import *
from math import floor
from scipy.signal import stft
from sklearn.utils import shuffle
from train_test_scripts import fit_predict
from vit_pytorch.cait import CaiT
from misc.utils import *

print("Please wait, data extraction in progress...")
# 1.) DEFINE SUBJECTS AND MOTOR IMAGERY PERIOD
fs = 250
start = 3 * fs
stop = 7 * fs


subjects = range(1,10)
# 2.) LOAD TIME SERIES
# print("Loading Time Series ...")
x_train, y_train = get_trials(subjects, start=start, stop=stop, dataset="train")
x_test, y_test = get_trials(subjects, start=start, stop=stop, dataset="eval")
x_train = x_train.swapaxes(1, 2)  # swap axes because STFT is computed over the last axis
x_test = x_test.swapaxes(1, 2)
# x_train.shape is (3026, 3, 1000)
# x_test.shape is (2241, 3, 1000)

results = []



for window_size, overlap in [(128,120), (256, 250), (512, 500) ]:
    for segment_size in [2, 5]:
        # 3.) SHUFFLE DATA AND SPLIT
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)

        # 4.) TRANSFORM TO FREQUENCY DOMAIN
        f_train, t_train, x_train_stft = stft(x_train, nperseg=window_size, noverlap=overlap, fs=250)
        f_test, t_test, x_test_stft = stft(x_test, nperseg=window_size, noverlap=overlap, fs=250)
        # Get the indices for the mu and beta bands
        mu_beta_band = np.where(np.logical_and(f_train >= 8, f_train <= 30))[0]

        Zxx_train = np.abs(x_train_stft[:, :, mu_beta_band, :])
        Zxx_test = np.abs(x_test_stft[:, :, mu_beta_band, :])



        required_padding = (segment_size - (Zxx_train.shape[3] % segment_size)) % segment_size
        Zxx_train = np.pad(Zxx_train, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant',
                           constant_values=0)
        Zxx_test = np.pad(Zxx_test, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)


        for dim in [256, 128]:
            for dropout in [0.1, 0.2]:
                for emb_dropout in [0.1, 0.2]:
                    for layer_dropout in [0.1, 0.2]:
                        v = CaiT(
                            image_height=Zxx_train.shape[2],
                            image_width=Zxx_train.shape[3],
                            patch_height=Zxx_train.shape[2],
                            patch_width=segment_size,
                            num_classes=2,
                            dim=dim,
                            depth=6, #not super important according to Random Forest
                            cls_depth=2,  # depth of cross attention of CLS tokens to patch
                            heads=16, #not super important according to Random Forest
                            mlp_dim=256, #not super important according to Random Forest
                            dropout=dropout,
                            emb_dropout=emb_dropout,
                            layer_dropout = layer_dropout
                        )

                    test_acc = fit_predict(v, Zxx_train, y_train, Zxx_test, y_test, epochs=10, crossval=10)
                    print("Test Accuracy:", test_acc)
                    result = [window_size, overlap, segment_size, dim, dropout, emb_dropout, layer_dropout, test_acc]
                    results.append(result)

# Save the results to a CSV file
sorted_results = sort_results_by_accuracy(results)
header = ["Window_Size", "Overlap", "Segment_Size", "Dim", "Dropout", "Emb_Dropout", "Layer_Dropout", "Test_Accuracy"]
save_results_to_csv("Results/STFT.csv", header, sorted_results)