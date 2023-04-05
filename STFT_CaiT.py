from sklearn.model_selection import train_test_split
import numpy as np
from misc.Visualization import *
from scipy.signal import stft
from sklearn.utils import shuffle
from train_test_scripts import train, test
from vit_pytorch.cait import CaiT
from misc.utils import get_trials

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

# 3.) TRANSFORM THE TIME SERIES INTO THE FREQUENCY DOMAIN WITH STFT
# print("Transforming to Frequency Domain ...")
WIN_SIZE = 512
f_train, t_train, Zxx_train = stft(x_train, nperseg=WIN_SIZE, noverlap=500, fs=fs)
f_test, t_test, Zxx_test = stft(x_test, nperseg=WIN_SIZE, noverlap=500, fs=fs)
Zxx_train = np.abs(Zxx_train)  # Neglect phase information
Zxx_test = np.abs(Zxx_test)

plt_avg_spectra_diff_stft(f_test, t_test, Zxx_test, y_test, freq_range=(8, 30), window_size=512, subjects=subjects)


# Get the indices for the mu and beta bands
mu_indices = np.where(np.logical_and(f_train >= 4, f_train <= 15))[0]
beta_indices = np.where(np.logical_and(f_train >= 19, f_train <= 30))[0]

Zxx_train = Zxx_train[:, :, np.concatenate([mu_indices, beta_indices]), :]
Zxx_test = Zxx_test[:, :, np.concatenate([mu_indices, beta_indices]), :]

# 4.) VISUALIZE THE DATA


# 5.) SHUFFLE DATA AND SPLIT
x_train, y_train = shuffle(Zxx_train, y_train)
x_test, y_test = shuffle(Zxx_test, y_test)


# 6.) DEFINE TRANSFORMER
# print("Training begins ...")
v = CaiT(
    image_size=max(x_train.shape[2:]),
    patch_size=5,
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

best_model = train(v, x_train, y_train, epochs=10, crossval=3)
test_acc = test(best_model, x_test, y_test)
print("Test Accuracy:", test_acc)
print("----------------------------------------------")
print("COHEN_KAPPA: ", (test_acc - 1/2)/0.5)