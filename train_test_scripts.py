import torch
import torch.nn as nn
import torch.optim as optim
from math import ceil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import Tensor
from copy import deepcopy

def fit_predict(v_start, x_train, y_train, x_test, y_test, epochs, crossval, fancy_plot=True):
    lr = 3e-5
    batch_size = 64
    criterion = nn.CrossEntropyLoss()
    total_test_accuracy = 0

    if fancy_plot:
        fig, axs = plt.subplots(2, crossval, sharex="all", sharey="all")
        fig.set_figheight(4)
        fig.set_figwidth(4 * crossval)

    for k in range(crossval):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train)
        batches_train = ceil((x_train.shape[0] / batch_size))
        batches_val = ceil((x_val.shape[0] / batch_size))
        batches_test = ceil((x_test.shape[0] / batch_size))

        v = deepcopy(v_start).to("cuda")
        optimizer = optim.Adam(v.parameters(), lr=lr)

        train_acc_history = []
        val_acc_history = []
        train_loss_history = []
        val_loss_history = []

        highest_observed_val_acc = 0
        best_model = None

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            for batch_index in range(batches_train):
                data = Tensor(x_train)[batch_index * batch_size:(batch_index + 1) * batch_size]
                label = Tensor(y_train)[batch_index * batch_size: (batch_index + 1) * batch_size].long() - 1
                data = data.to("cuda")
                label = label.to("cuda")

                output = v(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / batches_train
                epoch_loss += loss / batches_train

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0

                for batch_index in range(batches_val):
                    data = Tensor(x_val)[batch_index * batch_size: (batch_index + 1) * batch_size]
                    label = Tensor(y_val)[batch_index * batch_size: (batch_index + 1) * batch_size].long() - 1
                    data = data.to("cuda")
                    label = label.to("cuda")

                    val_output = v(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / batches_val
                    epoch_val_loss += val_loss / batches_val

            if epoch_val_accuracy > highest_observed_val_acc:
                highest_observed_val_acc = epoch_val_accuracy
                best_model = deepcopy(v)

            train_acc_history.append(float(epoch_accuracy))
            val_acc_history.append(float(epoch_val_accuracy))
            train_loss_history.append(float(epoch_loss))
            val_loss_history.append(float(epoch_val_loss))

        if fancy_plot:
            axs[0, k].plot(train_acc_history)
            axs[0, k].plot(val_acc_history)
            axs[0, k].set_title(f"Fold k = {k}")
            axs[0, 0].set_ylabel("% Accuracy")
            axs[0, k].legend(["Train", "Val"])
            axs[0, k].set(adjustable='box')
            axs[1, k].set_xlabel("# Epochs")
            axs[1, 0].set_ylabel("Loss")
            axs[1, k].plot(train_loss_history)
            axs[1, k].plot(val_loss_history)
        # Test the best model on the test set for the current fold
        epoch_test_accuracy = 0
        for batch_index in range(batches_test):
            data = Tensor(x_test)[batch_index * batch_size: (batch_index + 1) * batch_size]
            label = Tensor(y_test)[batch_index * batch_size: (batch_index + 1) * batch_size].long() - 1
            data = data.to("cuda")
            label = label.to("cuda")

            test_output = best_model(data)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / batches_test

        total_test_accuracy += epoch_test_accuracy / crossval

    if fancy_plot:
        plt.tight_layout()
        plt.show()

    return float(total_test_accuracy)