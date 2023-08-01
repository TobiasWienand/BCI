import pywt
from scipy.signal import stft
from train_test_scripts import *
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
x_train = x_train.swapaxes(1, 2)
x_test = x_test.swapaxes(1, 2)


# 3.) TRANSFORM TO FREQUENCY DOMAIN
f_train_stft, t_train_stft, x_train_stft = stft(x_train, nperseg=512, noverlap=500, fs=250)
f_test_stft, t_test_stft, x_test_stft = stft(x_test, nperseg=512, noverlap=500, fs=250)
# Get the indices for the mu and beta bands
mu_beta_band = np.where(np.logical_and(f_train_stft >= 8, f_train_stft <= 30))[0]

# Apply Bandpass and get Power Spectral Density
Zxx_train = np.abs(x_train_stft[:, :, mu_beta_band, :])**2
Zxx_test = np.abs(x_test_stft[:, :, mu_beta_band, :])**2

# Pad with zeros to make the PSD width divisible by the time segment size
required_padding = (2 - (Zxx_train.shape[3] % 2)) % 2
Zxx_train = np.pad(Zxx_train, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant',
                   constant_values=0)
Zxx_test = np.pad(Zxx_test, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)

# Grid search the appropriate dimension and dropout for the data format

v_STFT = CaiT(
    image_height=Zxx_train.shape[2],
    image_width=Zxx_train.shape[3],
    patch_height=Zxx_train.shape[2],
    patch_width=2,
    num_classes=2,
    dim=128,
    depth=6, #not super important according to Random Forest
    cls_depth=2,  # depth of cross attention of CLS tokens to patch. Unique to CaiT
    heads=16, #not super important according to Random Forest
    mlp_dim=256, #not super important according to Random Forest
    dropout=0.1,
    emb_dropout=0.1,
    layer_dropout = 0.2 # Unique to CaiT. Triple regularization
)
total_acc, confidences_stft = fit_predict_confidences(v_STFT, Zxx_train, y_train, Zxx_test, y_test, epochs=10, crossval=10)
print(f"STFT accuracy: {total_acc}")
#########


x_train_cwt, scales_train = pywt.cwt(x_train, pywt.frequency2scale("cmor2.0-1.0", np.arange(8, 30)/fs), "cmor2.0-1.0")
x_test_cwt, scales_test = pywt.cwt(x_test, pywt.frequency2scale("cmor2.0-1.0", np.arange(8, 30)/fs), "cmor2.0-1.0")
x_train_cwt = x_train_cwt.transpose(1, 2, 0, 3)
x_test_cwt = x_test_cwt.transpose(1, 2, 0, 3)

x_train_cwt = densify(x_train_cwt, 10)
x_test_cwt = densify(x_test_cwt, 10)


if x_train_cwt.dtype == complex:
    x_train_cwt = np.abs(x_train_cwt)
    x_test_cwt = np.abs(x_test_cwt)

# Pad with zeros to make the PSD width divisible by the time segment size
required_padding = (2 - (x_train_cwt.shape[3] % 2)) % 2
x_train_cwt = np.pad(x_train_cwt, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)
x_test_cwt = np.pad(x_test_cwt, ((0, 0), (0, 0), (0, 0), (0, required_padding)), mode='constant', constant_values=0)

# Grid search the appropriate dimension and dropout for the data format

v_CWT = CaiT(
    image_height=x_train_cwt.shape[2],
    image_width=x_train_cwt.shape[3],
    patch_height=x_train_cwt.shape[2],
    patch_width=2,
    num_classes=2,
    dim=128,
    depth=6, #not super important according to Random Forest
    cls_depth=2,  # depth of cross attention of CLS tokens to patch. Unique to CaiT
    heads=16, #not super important according to Random Forest
    mlp_dim=256, #not super important according to Random Forest
    dropout=0.2,
    emb_dropout=0.1,
    layer_dropout = 0.2 # Unique to CaiT. Triple regularization
)

total_acc, confidences_wavelet = fit_predict_confidences(v_CWT, x_train_cwt, y_train, x_test_cwt, y_test, epochs=10, crossval=10)
print(f"Shannon wavelet accuracy: {total_acc}")
# Calculate combined confidences
combined_confidences = confidences_stft + confidences_wavelet

# Make predictions based on the class with greatest sum confidence
predictions_combined = np.argmax(combined_confidences, axis=1) + 1

# Compare with actual labels to calculate accuracy
accuracy_combined = (predictions_combined == y_test).mean()

print(f"Ensemble accuracy: {accuracy_combined}")