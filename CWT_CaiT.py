from train_test_scripts import fit_predict
from vit_pytorch.cait import CaiT
from misc.utils import *
from misc.Visualization import visualize
import numpy as np
import pywt


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
x_train = x_train.swapaxes(1, 2)
x_test = x_test.swapaxes(1, 2)


results = []
# 3.) TRANSFORM TO FREQUENCY DOMAIN
for wavelet in ["morl", "mexh"]:
    for density in [10, 20]:
        for segment_size in [2, 5]:

            x_train_cwt, freqs_train = pywt.cwt(x_train, np.arange(8, 30), wavelet)
            x_test_cwt, freqs_test = pywt.cwt(x_test, np.arange(8, 30), wavelet)
            x_train_cwt = x_train_cwt.transpose(1, 2, 0, 3)
            x_test_cwt = x_test_cwt.transpose(1, 2, 0, 3)

            x_train_cwt = densify(x_train_cwt, density)
            x_test_cwt = densify(x_test_cwt, density)

            # Pad with zeros to make the PSD width divisible by the time segment size
            required_padding = (segment_size - (x_train_cwt.shape[3] % segment_size)) % segment_size
            x_train_cwt = np.pad(x_train_cwt, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)
            x_test_cwt = np.pad(x_test_cwt, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)
            visualize(x_train_cwt, np.array([3+ i/250 for i in range(x_train_cwt.shape[-1]*density)]), freqs_train, y_train, "CWT")

            # Grid search the appropriate dimension and dropout for the data format
            for dim in [256, 128]:
                for dropout in [0.1, 0.2]:
                    for emb_dropout in [0.1, 0.2]:
                        for layer_dropout in [0.1, 0.2]:
                            v = CaiT(
                                image_height=x_train_cwt.shape[2],
                                image_width=x_train_cwt.shape[3],
                                patch_height=x_train_cwt.shape[2],
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

                            total_acc, individual_accs = fit_predict(v, x_train_cwt, y_train, x_test_cwt, y_test, IDs, epochs=10, crossval=10)
                            print("Total Test Accuracy:", total_acc)
                            result = [wavelet, density, segment_size, dim, dropout, emb_dropout, layer_dropout, total_acc]
                            # Add individual accuracies to the result list
                            for subject_id in sorted(individual_accs.keys()):
                                result.append(individual_accs[subject_id])

                            results.append(result)

# Save the results to a CSV file
header = ["Wavelet", "Density", "Segment_Size", "Dim", "Dropout", "Emb_Dropout", "Layer_Dropout", "Total_Test_Accuracy"] + [f"Subject_{i}_Accuracy" for i in range(1, 10)]
save_results_to_csv("Results/CWT.csv", header, results)