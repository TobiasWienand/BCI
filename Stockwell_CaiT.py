from train_test_scripts import fit_predict
from vit_pytorch.cait import CaiT
from misc.utils import *
import numpy as np
from misc.Visualization import visualize
from stockwell import st
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
x_train_transformed = np.apply_along_axis(lambda arr: st.st(arr, 8, 30), 2, x_train)
x_test_transformed = np.apply_along_axis(lambda arr: st.st(arr, 8, 30), 2, x_test)
#np.save("Data/x_train_transformed.npy", x_train_transformed)
#np.save("Data/x_test_transformed.npy", x_test_transformed)
#x_train_transformed = np.load("Data/x_train_transformed.npy") #Unfortunately I can only use Torch in Python Version 3.11 and Stockwell in 3.7
#x_test_transformed = np.load("Data/x_test_transformed.npy")

results = []
# 3.) TRANSFORM TO FREQUENCY DOMAIN
for density in [10, 20]:
    for segment_size in [2, 5]:
        # Densify the data
        densified_train = densify(x_train_transformed, density)
        densified_test = densify(x_test_transformed, density)

        # Create separate arrays for the real and imaginary parts
        x_train_transformed_dense = np.concatenate((densified_train.real, densified_train.imag), axis=2)
        x_test_transformed_dense = np.concatenate((densified_test.real, densified_test.imag), axis=2)


        # Pad with zeros to make the PSD width divisible by the time segment size
        required_padding = (segment_size - (x_train_transformed_dense.shape[3] % segment_size)) % segment_size
        x_train_transformed_dense = np.pad(x_train_transformed_dense, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)
        x_test_transformed_dense = np.pad(x_test_transformed_dense, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)
        visualize(x_train_transformed_dense, np.array([3 + i / 250 for i in range(x_train_transformed_dense.shape[-1]*density)]), np.array(range(8,31)), y_train,
                  "Stockwell")

        # Grid search the appropriate dimension and dropout for the data format
        for dim in [256, 128]:
            for dropout in [0.1, 0.2]:
                for emb_dropout in [0.1, 0.2]:
                    for layer_dropout in [0.1, 0.2]:
                        v = CaiT(
                            image_height=x_train_transformed_dense.shape[2],
                            image_width=x_train_transformed_dense.shape[3],
                            patch_height=x_train_transformed_dense.shape[2],
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

                        total_acc, individual_accs = fit_predict(v, x_train_transformed_dense, y_train, x_test_transformed_dense, y_test, IDs, epochs=10, crossval=10)
                        print("Total Test Accuracy:", total_acc)
                        result = [density, segment_size, dim, dropout, emb_dropout, layer_dropout, total_acc]
                        # Add individual accuracies to the result list
                        for subject_id in sorted(individual_accs.keys()):
                            result.append(individual_accs[subject_id])

                        results.append(result)

# Save the results to a CSV file
header = ["Density", "Segment_Size", "Dim", "Dropout", "Emb_Dropout", "Layer_Dropout", "Total_Test_Accuracy"] + [f"Subject_{i}_Accuracy" for i in range(1, 10)]
save_results_to_csv("Results/Stockwell.csv", header, results)