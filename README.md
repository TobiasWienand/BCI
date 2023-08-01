# Results at a glance

|                                                                              |                                                                                |                                                                                 |
|:----------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|
|           <img width="1604"  src="Results/img/STFT.png">  STFT 76%           |             <img width="1604"  src="Results/img/CWT_shan.png">  Complex Shannon Wavelet 71%            |        <img width="1604"  src="Results/img/CWT_cmor.png">  Complex Morl Wavelet 71%         |
|            <img width="1604"  src="Results/img/HHT.png">  HHT 69%            |             <img width="1604"  src="Results/img/CWT_cgau6.png">  Complex Gauss Wavelet 69%             |        <img width="1604"  src="Results/img/Stockwell.png">Stockwell 66% |

Ensemble method: 77%
# Description of the Project Structure


## Data

The data directory contains `.mat` files which consist of the data for 9 subjects. For each subject, there are two files: one for training and one for evaluation. More information about the data is available [here](https://www.bbci.de/competition/iv/desc_2b.pdf)

## misc/utils

Contains various utility functions, most notably the get_trials function that serves to extract the EEG data from the .mat files while removing artefacts, selecting subjects and only extracting a specified time segment. For more utility functions, please visit the file

## misc/Visualization

Contains a function that creates a plot with 3 subplots, each of which highlights the difference of the average spectro- or scalogram for class 1 and class 2 for a channel. If we see bright colors that means there is a stark difference between class 1 and class 2, indicating good classifyability

## Results/img

The aforementioned plots

## Results/*.csv

Contains csv files with the experimental results

## torchHHT

A repository that implements the Hilbert Huang Transformation using Torch

## *_CaiT.py

There are several files in the main directory that look like <name of method>_CaiT.py. Each of these implements the analysis
procedure for a specific time frequency analysis method. First the data is loaded and then a grid search is performed to find
suitable parameters for the specific method (e.g. overlap for STFT, wavelet for CWT and so on) and the optimal parameters for
the classifier, in our case the CaiT Vision Transformer. The accuracy and variance in general and for each subject are cross validated
10-fold and saved in a csv file. The training is done for 10 epochs. 

## train_test_scripts.py

The fit_predict function trains the neural network, predicts the class labels on the test data and returns the results. Optionally the training is plotted.
The fit_predict_confidences function works mostly the same but without visualization and logging. At the end, the confidences are extracted from the last (softmax) layer

## Ensemble.py

This implements an ensemble model, incorporating two Vision Transformer, each of which has been trained on either STFT or Shannon Wavelet transformed data. 
Using the confidences from both models, the class is predicted that has the highest sum confidence