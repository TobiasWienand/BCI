import numpy as np
import matplotlib.pyplot as plt

def plt_avg_spectra_diff_stft(f, t, Zxx_all, Y, freq_range, window_size, subjects):
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

def plot_avg_spectra_left_right_stft(f, t, Zxx_all, Y, freq_range, window_size):
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

def visualize_preprocessing(x_train, x_train_filtered, y_train):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Preprocessing with 8-35 Hz BPF', fontsize=16)
    axs[0].plot(x_train[0, 0, :], label="x[0,0,:]")
    axs[0].plot(x_train_filtered[0, 0, :], label="x_filtered[0,0,:]")
    axs[0].set_title("Example Signal Before and After")
    axs[0].set_ylabel("V")
    axs[0].set_xlabel("t_i with fs = 250")
    axs[0].legend()

    y_train_label_1_mean = np.mean(x_train[y_train==1, 0, :], axis=0)
    y_train_label_2_mean = np.mean(x_train[y_train==2, 0, :], axis=0)
    y_train_filtered_label_1_mean = np.mean(x_train_filtered[y_train==1, 0, :], axis=0)
    y_train_filtered_label_2_mean = np.mean(x_train_filtered[y_train==2, 0, :], axis=0)

    axs[1].plot(y_train_label_1_mean - y_train_label_2_mean, label="Original")
    axs[1].plot(y_train_filtered_label_1_mean - y_train_filtered_label_2_mean, label="Filtered")
    axs[1].set_title("Differences between Mean Signals with Class Label 1 vs Class Label 2")
    axs[1].set_ylabel("V")
    axs[1].set_xlabel("t_i with fs = 250")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def visualize_cwt(x_train_cwt, y_train, start, stop, HI, LO):
    # Create 3 subplots
    f, axarr = plt.subplots(3, sharex=True, sharey=True)
    f.suptitle('Difference between Mean Spectra of Class 1 and 2', fontsize=16)
    axarr[0].set_title('C3')
    axarr[1].set_title('CZ')
    axarr[2].set_title('C4')
    # Iterate over all three channels

    diff_matrices = []
    for k in range(3):
        # Get mean values of each class
        class_1 = np.mean(x_train_cwt[y_train == 1, k, :, :], axis=0)
        class_2 = np.mean(x_train_cwt[y_train == 2, k, :, :], axis=0)
        # Calculate the difference
        difference = np.abs(class_1 - class_2)
        # Plot the difference
        axarr[k].imshow(difference, cmap=plt.cm.seismic, aspect='auto')

        # Set the y-axis ticks in Hz
        tick_locs = [0, x_train_cwt.shape[-2] / 2, x_train_cwt.shape[-2] - 1]
        tick_labels = [LO, (HI - LO) / 2, HI]
        axarr[k].set_yticks(tick_locs)
        axarr[k].set_yticklabels(tick_labels)

        diff_matrices.append(difference)

    # Set the y-axis label
    axarr[1].set_ylabel('f (Hz)')
    # Set the x-axis label
    axarr[2].set_xlabel('Time (s)')


    # Set the x-axis ticks in seconds
    tick_locs = np.arange(0, x_train_cwt.shape[-1], 250)
    tick_labels = np.arange(start/250, stop/250, 1)
    axarr[2].set_xticks(tick_locs)
    axarr[2].set_xticklabels(tick_labels)


    # Show the plot
    plt.tight_layout()
    plt.show()

    return diff_matrices[0], diff_matrices[1], diff_matrices[2]