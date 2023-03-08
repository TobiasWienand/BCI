from sklearn.model_selection import train_test_split
from matlab_data2python import get_trials
from scipy.signal import stft
from sklearn.utils import shuffle
import numpy as np
from ViT import train, test
from vit_pytorch.cait import CaiT
from misc.utils import *

print("Please wait, data extraction in progress...")
# 1.) DEFINE SUBJECTS AND MOTOR IMAGERY PERIOD
fs = 250
start = 3 * fs
stop = 7 * fs

cohen_kappas = np.zeros(10)
for subject in range(1, 10):
    print("TESTING SUBJECT", subject)
    # subjects = 4
    # 2.) LOAD TIME SERIES
    # print("Loading Time Series ...")
    x_train, y_train = get_trials(subject, start=start, stop=stop, dataset="train")
    x_test, y_test = get_trials(subject, start=start, stop=stop, dataset="eval")
    x_train = x_train.swapaxes(1, 2)  # swap axes because STFT is computed over the last axis
    x_test = x_test.swapaxes(1, 2)

    # 3.) TRANSFORM THE TIME SERIES INTO THE FREQUENCY DOMAIN WITH STFT
    # print("Transforming to Frequency Domain ...")
    _, _, Zxx_train = stft(x_train, nperseg=64, fs=fs)
    _, _, Zxx_test = stft(x_test, nperseg=64, fs=fs)
    Zxx_train = np.abs(Zxx_train)  # Neglect phase information
    Zxx_test = np.abs(Zxx_test)

    # 4.) SHUFFLE DATA AND SPLIT
    x_train, y_train = shuffle(Zxx_train, y_train)
    x_test, y_test = shuffle(Zxx_test, y_test)

    # 5.) DEFINE TRANSFORMER
    # print("Training begins ...")
    v = CaiT(
        image_size=33,
        patch_size=3,
        num_classes=2,
        dim=128,
        depth=6,
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=16,
        mlp_dim=256,
        dropout=0.2,
        emb_dropout=0.1,
        layer_dropout=0.05  # randomly dropout 5% of the layers
    )

    best_model = train(v, x_train, y_train, epochs=500, crossval=3)
    test_acc = test(best_model, x_test, y_test)
    cohen_kappas[subject] = (test_acc - 0.5)/0.5
    print("Test Accuracy:", test_acc)
    print("----------------------------------------------")
cohen_kappas[0] = np.mean(cohen_kappas[1:])
print("COHEN_KAPPAS: intra (mean, subject 1, ... subject 9):", cohen_kappas)