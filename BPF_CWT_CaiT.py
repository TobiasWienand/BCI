from scipy import signal
from misc.Visualization import *
from misc.utils import get_trials
import pywt
from sklearn.utils import shuffle
from vit_pytorch.cait import CaiT
from train_test_scripts import train, test


print("Data extraction in progress...")
# 1.1) DEFINE SUBJECTS AND MOTOR IMAGERY PERIOD
fs = 250
start = 3 * fs
stop = 7 * fs
subjects = range(1, 10)

# 1.2) LOAD TIME SERIES
x_train, y_train = get_trials(subjects, start=start, stop=stop, dataset="train")
x_test, y_test = get_trials(subjects, start=start, stop=stop, dataset="eval")
x_train = x_train.swapaxes(1, 2)
x_test = x_test.swapaxes(1, 2)


# 2) PREPROCESSING with a 8-35 Hz Bandpass Filter
# Create a bandpass filter
HI = 29
LO = 9
b, a = signal.butter(4, [LO, HI], fs=250, btype='band', output="ba")
x_train_filtered = signal.filtfilt(b, a, x_train) # Apply the filter to x_train and x_test
x_test_filtered = signal.filtfilt(b, a, x_test)
visualize_preprocessing(x_train, x_train_filtered, y_train)

# 3) FEATURE EXTRACTION WITH CWT and morl wavelet
print("CWT in progress...")
x_train_cwt, freqs_train = pywt.cwt(x_train_filtered, np.arange(LO, HI), "morl")
x_test_cwt, freqs_test = pywt.cwt(x_test_filtered, np.arange(LO, HI), "morl")
x_train_cwt = x_train_cwt.transpose(1, 2, 0, 3)
x_test_cwt = x_test_cwt.transpose(1, 2, 0, 3)

visualize_cwt(x_train_cwt, y_train, start, stop, HI, LO)

x_train, y_train = shuffle(x_train_cwt, y_train)
x_test, y_test = shuffle(x_test_cwt, y_test)

# 6.) DEFINE TRANSFORMER
# print("Training begins ...")
v = CaiT(
    image_size=max(x_train.shape[2:]),
    patch_size=20,
    num_classes=2,
    dim=128,
    depth=6,
    cls_depth=2,  # depth of cross attention of CLS tokens to patch
    heads=16,
    mlp_dim=256,
    dropout=0.3,
    emb_dropout=0.2,
    layer_dropout=0.1  # randomly dropout 5% of the layers
)

best_model = train(v, x_train, y_train, epochs=10, crossval=3)
test_acc = test(best_model, x_test, y_test)
print("Test Accuracy:", test_acc)
print("----------------------------------------------")
print("COHEN_KAPPA: ", (test_acc - 1/2)/0.5)