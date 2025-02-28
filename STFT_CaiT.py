from scipy.signal import stft
from train_test_scripts import fit_predict
from vit_pytorch.cait import CaiT
from misc.utils import *
from misc.Visualization import visualize

print("Please wait, data extraction in progress...")
# 1.) DEFINE SUBJECTS AND MOTOR IMAGERY PERIOD
fs = 250
start = 3 * fs
stop = 7 * fs


subjects = range(1,10)
# 2.) LOAD TIME SERIES
# print("Loading Time Series ...")
x_train, y_train, _ = get_trials(subjects, start=start, stop=stop, dataset="train")
x_test, y_test, IDs = get_trials(subjects, start=start, stop=stop, dataset="eval")
x_train = x_train.swapaxes(1, 2)  # swap axes because STFT is computed over the last axis
x_test = x_test.swapaxes(1, 2)


results = []
for window_size, overlap in [(512, 500), (128, 64)]:
    for segment_size in [2, 5]:

        # 3.) TRANSFORM TO FREQUENCY DOMAIN
        f_train, t_train, x_train_stft = stft(x_train, nperseg=window_size, noverlap=overlap, fs=250)
        f_test, t_test, x_test_stft = stft(x_test, nperseg=window_size, noverlap=overlap, fs=250)
        # Get the indices for the mu and beta bands
        mu_beta_band = np.where(np.logical_and(f_train >= 8, f_train <= 30))[0]

        # Apply Bandpass and get Power Spectral Density
        Zxx_train = np.abs(x_train_stft[:, :, mu_beta_band, :])**2
        Zxx_test = np.abs(x_test_stft[:, :, mu_beta_band, :])**2

        # Pad with zeros to make the PSD width divisible by the time segment size
        required_padding = (segment_size - (Zxx_train.shape[3] % segment_size)) % segment_size
        Zxx_train = np.pad(Zxx_train, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant',
                           constant_values=0)
        Zxx_test = np.pad(Zxx_test, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)

        visualize(Zxx_train, np.array([3 + i / 250 for i in range(Zxx_train.shape[-1])]), f_train[mu_beta_band],
                  y_train, "STFT")
        # Grid search the appropriate dimension and dropout for the data format
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
                            cls_depth=2,  # depth of cross attention of CLS tokens to patch. Unique to CaiT
                            heads=16, #not super important according to Random Forest
                            mlp_dim=256, #not super important according to Random Forest
                            dropout=dropout,
                            emb_dropout=emb_dropout,
                            layer_dropout = layer_dropout # Unique to CaiT. Triple regularization
                        )

                        total_acc, individual_accs, total_var, individual_vars = fit_predict(v, Zxx_train, y_train, Zxx_test, y_test, IDs, epochs=10, crossval=10)
                        print("Total Test Accuracy:", total_acc)
                        result = [window_size, overlap, segment_size, dim, dropout, emb_dropout, layer_dropout, total_acc, total_var]
                        # Add individual accuracies to the result list
                        for subject_id in sorted(individual_accs.keys()):
                            result.append(individual_accs[subject_id])
                        for subject_id in sorted(individual_vars.keys()):
                            result.append(individual_vars[subject_id])

                        results.append(result)

# Save the results to a CSV file
header = ["Window_Size", "Overlap", "Segment_Size", "Dim", "Dropout", "Emb_Dropout", "Layer_Dropout", "Total_Test_Accuracy", "Total_Test_Variance"] + [f"Subject_{i}_Accuracy" for i in range(1, 10)] + [f"Subject_{i}_Variance" for i in range(1, 10)]
save_results_to_csv("Results/STFT.csv", header, results)