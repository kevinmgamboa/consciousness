"""
Created on Wed Jul 14 11:52:11 2021
@author: Kevin Machado Gamboa
#################################
Util functions such as
    1) plotting training history
    2) plotting confusion matrix
    3) plot image samples from dataset
#################################
"""
import os
import random as rd
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing personal libraries
import config
import databases as dbs
import functions_main as mpf

def plot_train_test_history(model_best):
    """
    @param model_best: dictionary with model information
    @return: Plots for history
    """
    fig = plt.figure()
    # Converts train history into DataFrame
    train_history = pd.DataFrame(model_best['train_history'][np.argmax(np.array(model_best['score']))],
                                 columns=['loss', 'acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])
    # Line stiles
    styles_acc, styles_loss = ['b', 'y.', 'r--'], ['b', 'y.', 'r--']
    ax1 = fig.add_subplot(211)
    # plots train history
    train_history[['acc', 'val_acc', 'test_acc']].plot(title='Accuracy Train-Val-Test',
                                                       style=styles_acc, linewidth=1.0,
                                                       grid=True, ax=ax1)
    ax2 = fig.add_subplot(212)
    # plots loss history
    train_history[['loss', 'val_loss', 'test_loss']].plot(title='Loss Train-Val-Test',
                                                          style=styles_acc, linewidth=1.0,
                                                          grid=True, ax=ax2)
    plt.xlabel('Epochs')
    plt.show()

def print_cross_validation_scores(test_acc_per_fold, test_loss_per_fold):
    """
    Provides a summary of scores after k-fold cross validation
    @param test_acc_per_fold: list of accuracy scores
    @param test_loss_per_fold: list of loss scores
    @return: None
    """
    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(test_acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Test Loss: {test_loss_per_fold[i]:.4} - Test Accuracy: {test_acc_per_fold[i]:.4}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Mean Test Accuracy: {np.mean(test_acc_per_fold):.4} (+- {np.std(test_acc_per_fold):.4})')
    print(f'> Mean Loss: {np.mean(test_loss_per_fold):.4}')
    print('------------------------------------------------------------------------')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    @param cm:
    @param classes:
    @param normalize:
    @param title:
    @param cmap:
    @return:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', size=12, fontweight='bold')
    plt.xlabel('Predicted label', size=12, fontweight='bold')
    plt.show()


def imshow_samples(dataset, classes, bar=False):
    """
    show image samples from the dataset
    @param dataset: dataset transformed
    @param classes: the vector of classes
    @return: shows the figure
    """
    # extract number of classes (nc)
    nc = len(classes)
    # defines the figure subplots
    fig, ax = plt.subplots(nrows=5, ncols=nc, figsize=(15, 10))
    # class selector
    cs = 0
    # Specification for barcode plot
    if bar:
        barprops = dict(aspect='auto', cmap='binary', interpolation='nearest')

    for row in ax:
        for col in row:
            # selects a random category (the index)
            sample_idx = rd.sample(list(np.where(dataset['labels'] == (cs % nc))[0]), 1)[0]

            # shows epoch
            if bar:
                # selects the epoch corresponded to category above
                spec_chunk = dataset['epochs'][sample_idx]
                col.imshow(spec_chunk, **barprops)
            else:
                # selects the epoch corresponded to category above
                spec_chunk = dataset['epochs'][sample_idx][:, :, 0]
                col.imshow(spec_chunk, aspect="auto", cmap='RdGy')
            # Increase class selector
            cs += 1
    # Creating labels per columns and rows
    cols = [classes[n] for n in classes]
    rows = ['Sample {}'.format(n) for n in range(1, 6)]

    # Adds titles per columns
    [axes.set_title(col) for axes, col in zip(ax[0], cols)]

    for axes, row in zip(ax[:, 0], rows):
        axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad - 5, 0),
                      xycoords=axes.yaxis.label, textcoords='offset points',
                      size='large', ha='right', va='center')

    fig.tight_layout()
    plt.show()

def show_samples(dataset, classes):
    """
    show image samples from the dataset
    @param dataset: dataset transformed
    @param classes: the vector of classes
    @return: shows the figure
    """
    # extract number of classes (nc)
    nc = len(classes)
    # defines the figure subplots
    fig, ax = plt.subplots(nrows=5, ncols=nc, figsize=(15, 10))
    # class selector
    cs = 0

    for row in ax:
        for col in row:
            # selects a random category (the index)
            sample_idx = rd.sample(list(np.where(dataset['labels'] == (cs % nc))[0]), 1)
            # selects the epoch corresponded to category above
            epoch = dataset['epochs'][sample_idx][0][0]
            # shows epoch
            col.plot(epoch)
            # Increase class selector
            cs += 1
    # Creating labels per columns and rows
    cols = [classes[n] for n in classes]
    rows = ['Sample {}'.format(n) for n in range(1,6)]

    # Adds titles per columns
    [axes.set_title(col) for axes, col in zip(ax[0], cols)]

    for axes, row in zip(ax[:,0], rows):
        axes.annotate(row,xy=(0, 0.5), xytext=(-axes.yaxis.labelpad-5, 0),
                      xycoords=axes.yaxis.label, textcoords='offset points',
                      size='large', ha='right', va='center')

    fig.tight_layout()
    plt.show()

def create_date():
    """
    Creates a date including year-day-time(h:m:s)
    @return: string with date
    """
    return str(datetime.now()).replace(' ', '_').replace('-', '').replace(':', '')[:-7]

def check_benchmark(model_best, database='sleep'):
    """

    @param model_best: dictionary with model information
    @param database: (str) either 'sleep' or 'anaesthesia'
    @return: None
    """
    # reads benchmark list
    bl = pd.read_csv('savings/benchmarks_' + database + '.csv', header=0)
    # compares model outcome with benchmark
    cm = bl.value_best_achieved.max()  #, bl.value_best_achieved.idxmax()
    # and bl.in_data_form[idx]
    # get the index for model with best score
    index = np.argmax(np.array(model_best['score']))
    # get the best score and multiplied to compare it
    best_score = int(np.array(model_best['score'])[index] * 10000)
    # stores the model in the benchmark
    if best_score > cm:
        print(f'New Benchmark Found .. \nold: {cm}\nnew: {best_score}')
        # date for logs
        date = str(datetime.now()).replace(' ', '_').replace('-', '').replace(':', '')[:-7]
        # creates file name
        export_dir = 'savings/' + str(int(best_score)) + '_' + database + '_' + \
                     date + '_' + model_best['transformation'] + '_' + model_best['name']
        # makes a folder
        os.mkdir(export_dir)
        # saving the model
        model_best['model'][index].save(export_dir + '/' + export_dir[8:] + '_model.h5')
        # filling columns for logs
        nb = [[export_dir[8:], model_best['name'], 'accuracy', best_score, config.NUM_FOLDS,
               "{:.4f}".format(np.mean(model_best['test_acc_per_fold'])),
               "{:.4f}".format(np.std(model_best['test_acc_per_fold'])),
               "{:.4f}".format(np.mean(model_best['test_loss_per_fold'])),
               model_best['transformation']]]
        # convert column into Dataframe
        bl = pd.DataFrame(nb)
        # updating benchmark.csv
        bl.to_csv('savings/benchmarks_' + database + '.csv', index=False, mode='a', header=False)
        # saving segments parameters for reproducibility
        with open(export_dir + '/' + export_dir[8:] + '_data_segments.npy', 'wb') as f:
            np.save(f, model_best['tra_with'][index])
            np.save(f, model_best['val_with'][index])
        # saving initial weights for reproducibility
        with open(export_dir + '/' + export_dir[8:] + '_initial_weights_.npy', 'wb') as f:
            for _, wei in enumerate(model_best['initial_weights'][index]):
                np.save(f, wei)
        # Stores train history
        train_history = pd.DataFrame(model_best['train_history'][index],
                                     columns=['loss', 'acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])
        train_history.to_pickle(export_dir + '/' + export_dir[8:] + '_train_history.pkl')

def super_test(model, transform_func, dataset='sleep'):
    """
    @param model:
    @param dataset:
    @param transform_func:
    @return:
    """


    if dataset == 'sleep':
        # initialize sleep database
        sleep = dbs.sleep()
        # loads [x_epochs, y_labels]
        sleep.load_epochs_labels(t_files=5)
        # converts labels to [0=>conscious,5* 1=>unconscious]
        sleep.get_binary_labels()
        # Normalize the dataset between [-1,1]
        sleep.transform(mpf.nor_dataset)
        # applying dataset transformation e.g. 'spectrogram'
        sleep.transform(transform_func)
        # make dataset ready for training
        sleep.get_ready_for_training()
        ##############################
        print(model.evaluate(sleep.data['train']['epochs'], sleep.data['train']['labels']))
        print(model.evaluate(sleep.data['validation']['epochs'], sleep.data['validation']['labels']))
        print(model.evaluate(sleep.data['test']['epochs'], sleep.data['test']['labels']))

    elif dataset == 'anaesthesia':
        # initialize anaesthesia database
        anaesthesia = dbs.anaesthesia()
        # path to dataset
        ane_data_path = "datasets\Kongsberg_anesthesia_data\EEG_resampled_2split_100Hz"
        # loads [x_epochs, y_labels]
        anaesthesia.load_epochs_labels(ane_data_path, selected_channels=config.channels_sleep_montage,
                                       sleep_montage=True)
        # converts labels to [0=>conscious, 1=>unconscious]
        anaesthesia.get_binary_labels()
        # Normalize the dataset between [-1,1]
        anaesthesia.transform(mpf.nor_dataset)
        # applying dataset transformation e.g. 'spectrogram'
        anaesthesia.transform(transform_func)
        # make dataset ready for training
        anaesthesia.get_ready_for_training()
        ####################################
        print(model.evaluate(anaesthesia.data['train']['epochs'], anaesthesia.data['train']['labels']))
        print(model.evaluate(anaesthesia.data['test']['epochs'], anaesthesia.data['test']['labels']))
