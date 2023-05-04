# Roadmap
This README serves to illustrate the progress in this project.

:clock1: STFT.csv will be uploaded this week. Currently the GPU is working on full throttle to complete the grid search
## 1. Rectangular Patches for CaiT :heavy_check_mark:
We have implemented rectangular patches for the CaiT architecture, which allows us to analyze how ViTs perform in the frequency domain.

## 2. Per Subject Test Accuracy :heavy_check_mark:

We have calculated the test accuracy for each subject, providing us with a granular understanding of how well our models perform on individual cases.

## 3. Grid Search :heavy_check_mark:

We have employed grid search to optimize hyperparameters for each feature extraction method, ensuring that we make accurate statements about the quality of extracted features.

## 4. Feature Extraction Methods

We will investigate various feature extraction methods, including:

-   4.1 Short-Time Fourier Transform (STFT) :heavy_check_mark:
    
    -   Parameters: window size, overlap, ~~window type (e.g. hanning)~~
-   4.2 Hilbert-Huang Transform (HHT)
    
    -   Parameters: maxiter, nbsym, stop_fun
-   4.3 Continuous Wavelet Transform (CWT)
    
    -   Parameters: wavelet function (e.g., Daubechies, Haar, etc.), scales, method (fft vs conv)
-   4.4 Stockwell Transform (S-Transform)
    
    -   Parameters: gamma (trades frequency with time resolution), wave type (gauss/kazemi)
-   4.5 Discrete Wavelet Transform (DWT)
    
    -   Parameters: wavelet function, signal extension mode
-   4.6 (Optional) Wigner-Ville Distribution
    
    -   Python module: tftb.processing.WignerVilleDistribution (no tunable parameters)
  
## 5. Ensemble Methods

After obtaining the best parameters for each transformation, we will explore ensemble methods to improve the performance of our models:

-   5.1 Weighted Voting
    -   Implement weighted voting based on the probability distribution of each class predicted by the ViT models.

## 6. Review and Future Steps

-   We will have a meeting to review the intermediate results and plan the next steps, which may include different ensembling methods or moving to the time domain.
