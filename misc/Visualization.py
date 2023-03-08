import itertools
import numpy as np
from numpy import concatenate as cat
from scipy.io import loadmat
from scipy.signal import stft
import matplotlib.pyplot as plt

####### HARDCORE GLOBAL VARIABLES THAT DONT CHANGE #######
T = []
E = []
fs = 250 # sampling frequency
##########################################################

def plt_avg_spectra_diff(f, t, Zxx_all, Y, freq_range, window_size, subjects):
    """
    :param f: Vector of frequencies returned by STFT
    :param t: Vector of hanning window start points returned by STFT
    :param Zxx: Array of frequency over time values. Returned by STFT
    :param Y: Array of labels. (Ground truth from the dataset)
    :param freq_range: The frequency range which we want to plot. Example: freq_range = (4,15) to extract mu-band
    :param window_size: The size of the hanning window from the STFT
    :param subjects: list or int representing which subjects the data was collected from
    :return: Nothing. Plots the three channels (C4, Cz, C3) vertically
    """

    # Implements a bandpass to only display a certain frequency range
    f_start = np.where(f > freq_range[0])[0][0]
    f_stop = np.where(f < freq_range[1])[0][-1]
    f = f[f_start:f_stop + 1]
    Zxx_bandpassed = Zxx_all[:, :, f_start:f_stop + 1, :]

    # Split the datasets in left and right
    Zxx_abs = np.abs(Zxx_bandpassed)
    Zxx_left = Zxx_abs[Y == 1, :, :, :]
    Zxx_right = Zxx_abs[Y == 2, :, :, :]

    Zxx_left_mean = np.mean(Zxx_left, axis=0)
    Zxx_right_mean = np.mean(Zxx_right, axis=0)

    Zxx_diff = np.abs(Zxx_left_mean - Zxx_right_mean)

    # Figure will have 3 rows because of three channels (C4, Cz and C3) and 2 columns for left and right labels
    fig, axs = plt.subplots(3)

    fig.suptitle('Absolute Diff of Mean Spectra (Hz) Over Time (s).\nWindow='+str(window_size)+'. Freq_range=' + str(freq_range[0]) + "-" + str(
        freq_range[1]) + ". Subject ID="+str(subjects), fontsize=15)

    # C3 channel in the first row
    axs[0].set_ylabel("C3 Channel")
    axs[0].pcolormesh(t, f, Zxx_diff[0, :], shading='auto')

    # CZ channel in the middle row
    axs[1].set_ylabel("CZ Channel")
    axs[1].pcolormesh(t, f, Zxx_diff[1, :], shading='auto')

    # C4 channel in the bottom row
    axs[2].set_ylabel("C4 Channel")
    axs[2].pcolormesh(t, f, Zxx_diff[2, :], shading='auto')
    fig.tight_layout()

    plt.show()

def plot_avg_spectra_left_right(f, t, Zxx_all, Y, freq_range, window_size):
    """
    :param f: Vector of frequencies returned by STFT
    :param t: Vector of hanning window start points returned by STFT
    :param Zxx: Array of frequency over time values. Returned by STFT
    :param Y: Array of labels. (Ground truth from the dataset)
    :param freq_range: The frequency range which we want to plot. Example: freq_range = (4,15) to extract mu-band
    :return: Nothing. Plots the three channels (C4, Cz, C3) vertically
    """

    # Implements a bandpass to only display a certain frequency range
    f_start = np.where(f > freq_range[0])[0][0]
    f_stop = np.where(f < freq_range[1])[0][-1]
    f = f[f_start:f_stop+1]
    Zxx_bandpassed = Zxx_all[:,:,f_start:f_stop+1,:]

    # Split the datasets in left and right
    Zxx_abs = np.abs(Zxx_bandpassed)
    Zxx_left = Zxx_abs[Y == 1, :, :, :]
    Zxx_right = Zxx_abs[Y == 2, :, :, :]

    Zxx_left_mean = np.mean(Zxx_left, axis=0)
    Zxx_right_mean = np.mean(Zxx_right, axis=0)

    # Figure will have 3 rows because of three channels (C4, Cz and C3) and 2 columns for left and right labels
    fig, axs = plt.subplots(3, 2)

    fig.suptitle('Average Spectra over time (s).\nWindow='+str(window_size)+'. Freq_range='+str(freq_range[0])+"-"+str(freq_range[1])+" Hz", fontsize=17)

    # C3 channel in the first row
    axs[0, 0].set_ylabel("C3 Channel")
    axs[0, 0].set_title("Left Hand MI")
    axs[0, 0].pcolormesh(t, f, Zxx_left_mean[0,:], shading='auto')
    axs[0, 1].pcolormesh(t, f, Zxx_right_mean[0, :], shading='auto')
    axs[0, 1].set_title("Right Hand MI")

    # CZ channel in the middle row
    axs[1, 0].set_ylabel("CZ Channel")
    axs[1, 0].pcolormesh(t, f, Zxx_left_mean[1, :], shading='auto')
    axs[1, 1].pcolormesh(t, f, Zxx_right_mean[1, :], shading='auto')

    # C4 channel in the bottom row
    axs[2, 0].set_ylabel("C4 Channel")
    axs[2, 0].pcolormesh(t, f, Zxx_left_mean[2, :], shading='auto')
    axs[2, 1].pcolormesh(t, f, Zxx_right_mean[2, :], shading='auto')
    fig.tight_layout()

    plt.show()


def get_trials(subjects, start, stop, dataset):
    """
    For the given subjects, extract the EEG time series from all trials and all sessions.
    :param subjects: list or int, representing subjects to extract trials from
    :param start: f_s * t, e.g. 250*3 for cue start in Screening
    :param stop: f_s * t, e.g. 250 * 7.5 for feedback period stop in Smiley Feedback
    :param dataset: "training", "eval", or "both"
    :return: X (EEG array: trials X 3 X duration) and y (vector of labels, i.e. numbers)
    """
    if type(subjects) == int:
        subjects = [subjects]

    if dataset == "both":
        X_t, Y_t = get_trials(subjects, start, stop, "train")
        X_e, Y_e = get_trials(subjects, start, stop, "eval")
        return cat([X_t, X_e]) , cat([Y_t, Y_e])

    matlab_data = T if dataset == "train" else E
    subject_data = [matlab_data[subject_id-1] for subject_id in subjects] # -1 because matlab indexing starts with 1

    X_t = []
    Y_t = []

    for struct, session in itertools.product(subject_data, range(3 if dataset == "train" else 2)): # 3 sessions in train, only 2 in eval
        data = struct["data"][0][session]
        X = data['X'][0][0]
        y = data['y'][0][0]
        trial_times = data['trial'][0][0]
        artifacts = data['artifacts'][0][0]

        for i in range(len(trial_times)):
            if not artifacts[i]:
                X_t.append(X[trial_times[i, 0]-1+start:trial_times[i, 0]-1+stop][:,:3]) #Ignore EOG -> [:,:3]
                Y_t.append(y[i])
    return np.stack(X_t), np.stack(Y_t).ravel() # ravel => remove dimension of length 1


################################################################
############################# MAIN #############################
################################################################

# LOAD .MAT DATA
for i in range(1,10):
    E.append(loadmat(f"../Data/B0{str(i)}E.mat"))
    T.append(loadmat(f"../Data/B0{str(i)}T.mat"))


start = 3*fs
stop = 7*fs
duration = stop - start + 1

# EXTRACT IMAGERY PERIOD DATA FROM ALL TRIALS, SESSIONS AND SUBJECTS
for subject in range(1,10):
    X_time, Y = get_trials(subject, start=start, stop=stop, dataset="train")

    # TRANSFORM THE TIME SERIES INTO THE FREQUENCY DOMAIN WITH STFT
    # SWAPAXES BECAUSE STFT IS COMPUTED OVER THE LAST AXIS
    X_time = X_time.swapaxes(1, 2)

    for f_range in [(0, 30)]:
        for size in [128]:
            f, t, Zxx = stft(X_time, nperseg=size, fs=250)
            t += start/fs # stft assumes that X_time starts at t=0, which is not true
            plt_avg_spectra_diff(f, t, Zxx, Y, freq_range=f_range, window_size=size, subjects=subject)



#cA, cD = pywt.dwt(X_time, wavelet='db4')