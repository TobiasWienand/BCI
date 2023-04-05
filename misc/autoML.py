from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from misc.utils import get_trials
from scipy.signal import stft
from sklearn.utils import shuffle
import numpy as np
from train_test_scripts import train, test
from vit_pytorch.cait import CaiT
from tpot import TPOTClassifier

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
WIN_SIZE = 105
f_train, t_train, Zxx_train = stft(x_train, nperseg=WIN_SIZE, fs=fs)
f_test, t_test, Zxx_test = stft(x_test, nperseg=WIN_SIZE, fs=fs)
Zxx_train = np.abs(Zxx_train)  # Neglect phase information
Zxx_test = np.abs(Zxx_test)

# Get the indices for the mu and beta bands
mu_indices = np.where(np.logical_and(f_train >= 4, f_train <= 15))[0]
beta_indices = np.where(np.logical_and(f_train >= 19, f_train <= 30))[0]

Zxx_train = Zxx_train[:, :, np.concatenate([mu_indices, beta_indices]), :]
Zxx_test = Zxx_test[:, :, np.concatenate([mu_indices, beta_indices]), :]

# 4.) SHUFFLE DATA AND SPLIT
x_train, y_train = shuffle(Zxx_train, y_train)
x_test, y_test = shuffle(Zxx_test, y_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[3] * x_train.shape[1]* x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[3] * x_test.shape[1]* x_test.shape[2])
tpot = TPOTClassifier(verbosity=2, random_state=42, generations=4)
tpot.fit(x_train, y_train)


y_pred = tpot.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)