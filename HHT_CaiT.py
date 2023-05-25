from train_test_scripts import fit_predict
from vit_pytorch.cait import CaiT
from misc.utils import *
from torchHHT import hht
import torch
from misc.Visualization import visualize
import numpy as np
print("Please wait, data extraction in progress...")
# 1.) DEFINE SUBJECTS AND MOTOR IMAGERY PERIOD
fs = 250
start = 3 * fs
stop = 7 * fs


subjects = range(1, 10)
# 2.) LOAD TIME SERIES
# print("Loading Time Series ...")
x_train, y_train, _ = get_trials(subjects, start=start, stop=stop, dataset="train")
x_test, y_test, IDs = get_trials(subjects, start=start, stop=stop, dataset="eval")
x_train = torch.Tensor(x_train.swapaxes(1, 2)).cuda()
x_test = torch.Tensor(x_test.swapaxes(1, 2)).cuda()

# 3.) TRANSFORM TO FREQUENCY DOMAIN
imfs, imfs_env, imfs_freq = hht.hilbert_huang(x_train, fs, num_imf=3)
Zxx_train, t_train, f_train = hht.hilbert_spectrum(imfs_env, imfs_freq, fs, freq_lim=(8, 30), time_scale=1, freq_res=1)
imfs, imfs_env, imfs_freq = hht.hilbert_huang(x_test, fs, num_imf=3)
Zxx_test, t_test, f_test = hht.hilbert_spectrum(imfs_env, imfs_freq, fs, freq_lim=(8, 30), time_scale=1, freq_res=1)
Zxx_train, t_train  = Zxx_train.cpu().numpy().swapaxes(2, 3)[:, :, :, :-1], t_train + start/fs
Zxx_test, t_test = Zxx_test.cpu().numpy().swapaxes(2, 3)[:, :, :, :-1], t_test + start/fs

results = []

for density in [5, 10, 20]:
    for segment_size in [2, 5]:
        # Densify the data
        Zxx_train_densified = densify(Zxx_train, density)
        Zxx_test_densified = densify(Zxx_test, density)

        visualize(Zxx_train_densified, t_train, f_train, y_train, "HHT")

        # Pad with zeros to make the PSD width divisible by the time segment size
        required_padding = (segment_size - (Zxx_train_densified.shape[3] % segment_size)) % segment_size
        Zxx_train_densified = np.pad(Zxx_train_densified, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)
        Zxx_test_densified = np.pad(Zxx_test_densified, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)

        # Grid search the appropriate dimension and dropout for the data format
        for dim in [256, 128]:
            for dropout in [0.1, 0.2]:
                for emb_dropout in [0.1, 0.2]:
                    for layer_dropout in [0.1, 0.2]:
                        v = CaiT(
                            image_height=Zxx_train_densified.shape[2],
                            image_width=Zxx_train_densified.shape[3],
                            patch_height=Zxx_train_densified.shape[2],
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

                        total_acc, individual_accs = fit_predict(v, Zxx_train_densified, y_train, Zxx_test_densified, y_test, IDs, epochs=10, crossval=10)
                        print("Total Test Accuracy:", total_acc)
                        result = [density, segment_size, dim, dropout, emb_dropout, layer_dropout, total_acc]
                        # Add individual accuracies to the result list
                        for subject_id in sorted(individual_accs.keys()):
                            result.append(individual_accs[subject_id])

                        results.append(result)

# Save the results to a CSV file
header = ["Density", "Segment_Size", "Dim", "Dropout", "Emb_Dropout", "Layer_Dropout", "Total_Test_Accuracy"] + [f"Subject_{i}_Accuracy" for i in range(1, 10)]
save_results_to_csv("Results/HHT.csv", header, results)