from train_test_scripts import fit_predict
from vit_pytorch.cait import CaiT
from misc.utils import *
import numpy as np
from wavelets_pytorch.transform import WaveletTransformTorch   # PyTorch version


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

# 3.) TRANSFORM TO FREQUENCY DOMAIN

dt = 0.1         # sampling frequency
dj = 0.125       # scale distribution parameter
batch_size = 32  # how many signals to process in parallel


# Initialize wavelet filter banks (scipy and torch implementation)
wa_torch = WaveletTransformTorch(dt, dj, cuda=True)

# Performing wavelet transform (and compute scalogram)
Zxx_train = wa_torch.cwt(x_train)
Zxx_test = wa_torch.cwt(x_test)






results = []
segment_size = 3

# Pad with zeros to make the PSD width divisible by the time segment size
required_padding = (segment_size - (Zxx_train.shape[3] % segment_size)) % segment_size
Zxx_train = np.pad(Zxx_train, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)
Zxx_test = np.pad(Zxx_test, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)

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

                total_acc, individual_accs = fit_predict(v, Zxx_train, y_train, Zxx_test, y_test, IDs, epochs=10, crossval=10)
                print("Total Test Accuracy:", total_acc)
                result = [segment_size, dim, dropout, emb_dropout, layer_dropout, total_acc]
                # Add individual accuracies to the result list
                for subject_id in sorted(individual_accs.keys()):
                    result.append(individual_accs[subject_id])

                results.append(result)

# Save the results to a CSV file
header = ["Segment_Size", "Dim", "Dropout", "Emb_Dropout", "Layer_Dropout", "Total_Test_Accuracy"] + [f"Subject_{i}_Accuracy" for i in range(1, 10)]
save_results_to_csv("Results/HHT.csv", header, results)