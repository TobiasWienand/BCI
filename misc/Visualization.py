import numpy as np
import matplotlib.pyplot as plt

def visualize(Zxx_train, t_train, f_train, y_train, method):
    # Separate Zxx_train into two classes according to y_train
    Zxx_train_class1 = Zxx_train[y_train == 1]
    Zxx_train_class2 = Zxx_train[y_train == 2]

    # Compute mean of arrays w.r.t the number of samples
    Zxx_train_class1_mean = np.mean(Zxx_train_class1, axis=0)
    Zxx_train_class2_mean = np.mean(Zxx_train_class2, axis=0)

    # Compute the absolute difference between the two means
    Zxx_diff = np.abs(np.abs(Zxx_train_class1_mean) - np.abs(Zxx_train_class2_mean))

    # Creating the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Mapping channels to subplot titles
    channel_names = ['C3 Channel', 'Cz Channel', 'C4 Channel']

    # Plotting each channel
    for i in range(3):
        im = axs[i].imshow(Zxx_diff[i], aspect='auto', origin='lower', extent=[t_train.min(), t_train.max(), f_train.min(), f_train.max()])
        axs[i].set_title(channel_names[i], fontsize=16) # Adjust fontsize here
        axs[i].set_xlabel('Time', fontsize=14) # Adjust fontsize here
        axs[i].set_ylabel('Frequency', fontsize=14) # Adjust fontsize here

    fig.suptitle(f'Absolute difference of mean {method} spectra for two classes', fontsize=20) # Adjust fontsize here
    plt.tight_layout()
    plt.show()