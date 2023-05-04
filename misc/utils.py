import itertools
import numpy as np
from numpy import concatenate as cat
from scipy.io import loadmat
import math
import csv

####### HARDCODED GLOBAL VARIABLES THAT DONT CHANGE #######
fs = 250  # sampling frequency
##########################################################

def get_trials(subjects, start, stop, dataset):
    """
    For the given subjects, extract the EEG time series from all trials and all sessions.
    :param subjects: list or int, representing subjects to extract trials from
    :param start: f_s * t, e.g. 250*3 for cue start in Screening
    :param stop: f_s * t, e.g. 250 * 7.5 for feedback period stop in Smiley Feedback
    :param dataset: "training", "eval", or "both"
    :return: X (EEG array: trials X 3 X duration), y (vector of labels, i.e. numbers) and subject IDs (see y)
    """
    T = []
    E = []

    # LOAD .MAT DATA
    for i in range(1, 10):
        E.append(loadmat(f"Data/B0{str(i)}E.mat"))
        T.append(loadmat(f"Data/B0{str(i)}T.mat"))

    if type(subjects) == int:
        subjects = [subjects]

    if dataset == "both":
        X_t, Y_t = get_trials(subjects, start, stop, "train")
        X_e, Y_e = get_trials(subjects, start, stop, "eval")
        return cat([X_t, X_e]) , cat([Y_t, Y_e])

    matlab_data = T if dataset == "train" else E
    subject_data = [(matlab_data[subject_id-1], subject_id) for subject_id in subjects] # -1 because matlab indexing starts with 1

    X_t = []
    Y_t = []
    subject_labels = []

    for (struct, subject_id), session in itertools.product(subject_data, range(3 if dataset == "train" else 2)): # 3 sessions in train, only 2 in eval
        data = struct["data"][0][session]
        X = data['X'][0][0]
        y = data['y'][0][0]
        trial_times = data['trial'][0][0]
        artifacts = data['artifacts'][0][0]

        for i in range(len(trial_times)):
            if not artifacts[i]:
                X_t.append(X[trial_times[i, 0]-1+start:trial_times[i, 0]-1+stop][:,:3]) #Ignore EOG -> [:,:3]
                Y_t.append(y[i])
                subject_labels.append(subject_id)
    return np.stack(X_t), np.stack(Y_t).ravel(), np.stack(subject_labels) # ravel => remove dimension of length 1

def divisors(n):
    divs = [1]
    for i in range(2,int(math.sqrt(n))+1):
        if n%i == 0:
            divs.extend([i,int(n/i)])
    divs.extend([n])
    return list(set(divs))


def sort_results_by_accuracy(results):
    return sorted(results, key=lambda x: x[-1], reverse=True)

def save_results_to_csv(filename, header, results):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for result in results:
            writer.writerow(result)