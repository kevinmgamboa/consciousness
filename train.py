"""
Train a model on the Sleep Dataset
Created on Wed Jun  9 21:18:16 2021
@author: kevin machado gamboa
"""
# -----------------------------------------------------------------------------
#                           Libraries Needed
# -----------------------------------------------------------------------------
import os
import copy
import numpy as np
import pandas as pd

# Libraries for training process
from sklearn.model_selection import KFold
# ML library
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Personal Libraries
from helpers_and_functions import config, main_functions as mpf, utils
import modelhub as mh
import databases as dbs

# %%
# ------------------------------------------------------------------------------------
#                               Loading sleep dataset
# ------------------------------------------------------------------------------------
# initialize sleep database
sleep = dbs.sleep()
# loads [x_epochs, y_labels]
sleep.load_epochs_labels(t_files=5, n_test=0.30)
# converts labels to [0=>conscious,5* 1=>unconscious]
sleep.get_binary_labels()
# Normalize the dataset between [-1,1]
sleep.transform(mpf.nor_dataset)
# applying dataset transformation e.g. 'spectrogram'
sleep.transform(mpf.raw_chunks_to_spectrograms, name='spectrogram')
# make dataset ready for training
sleep.get_ready_for_training()

# %%
# ------------------------------------------------------------------------------------
#                                Cross-Validation
# ------------------------------------------------------------------------------------
# Creates folder to store experiment
date = utils.create_date()
os.mkdir('log_savings/sleep_' + date)
# confusion matrix per fold variable
cm_per_fold = []
# number of train epochs
train_epochs = 30
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
#                                               verbose=1, restore_best_weights=True)
kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True)
# init fold number
fn = 1

# stores the models with best acc & other info
model_best = {'model': [],
          'score': [],
          'tra_with': [],
          'val_with': [],
          'train_history': [],
          'initial_weights': [],
          'test_acc_per_fold': [],
          'test_loss_per_fold': [],
          'transformation': sleep.info['transformation']}

# training parameters
parameters = {'lr': 1e-6,
          'num_filters': 10,
          'kernel_size': 3,
          'dense_units': 10,
          'out_size': 1}

p_count = 0  # early stopping counter
patient = 5  # wait n epochs for error to keep decreasing, is not stop

all_folds_best_test_score = 0.0
for tra, val in kfold.split(sleep.data['train']['epochs'], sleep.data['train']['labels']):
    # Call the hub
    hub = mh.simple_cnn_2(param=parameters)
    # build model structure
    hub.build_model_structure(sleep.info['data_shape'])
    # compile model
    hub.compile()
    # initializing model
    model_best_fold = tf.keras.models.clone_model(hub.model)
    # initial model weights
    ini_wei = hub.model.get_weights()
    # initializing training history
    train_history = []
    # defines an initial score
    pre_score = [1.0, 0.0]
    # Generate a print
    print(100 * '-')
    print(f'--------------------------------- Training for fold {fn} ---------------------------------')
    # -----------------------------------------------------------------------------
    #                               Train-Test Loop
    # -----------------------------------------------------------------------------
    for n_ep in range(train_epochs):
        print('------- train score -------')
        # Train the model
        train_score = hub.model.pdf(sleep.data['train']['epochs'][tra],
                                    sleep.data['train']['labels'][tra],
                                    validation_data=(sleep.data['train']['epochs'][val],
                                                     sleep.data['train']['labels'][val]),
                                    epochs=1)#, callbacks=[early_stop])
        print('------- test score -------')
        # Evaluates on Test set
        test_scores = hub.model.evaluate(sleep.data['test']['epochs'], sleep.data['test']['labels'])
        # saving train history
        train_score = list(np.concatenate(list(train_score.history.values())))
        # Adding test score to train score
        train_score.extend(test_scores)
        # Train history including the score in test set
        train_history.append(train_score)

        # Stores the best model in the fold
        if test_scores[1] > pre_score[1]:
            print(f'new best score in the fold: {test_scores[1]:.4}')
            # saves best model INSIDE FOLD
            hub.model.save('log_savings/sleep_' + date + '/best_fold_model.h5')
            # Saves best model from ALL FOLDS
            if test_scores[1] > all_folds_best_test_score:
                print(f'new best model from ALL FOLDS {test_scores[1]:.4} ')
                all_folds_best_test_score = test_scores[1]
                # saves best model
                hub.model.save('log_savings/sleep_' + date + '/all_folds_best_model.h5')
            # updating previous score
            pre_score = copy.copy(test_scores)
            # reset the stopping patient counter
            p_count = 0
        else:  # Stopping criteria:
            p_count += 1
            if p_count >= patient:
                print('Early Stopping !!! Error hasnt decreased')
                p_count = 0
                break
    # -----------------------------------------------------------------------------
    #                          Stores Data from Each Fold
    # -----------------------------------------------------------------------------
    # save train history
    model_best['train_history'].append(train_history)
    # save best score from fold
    train_history = pd.DataFrame(train_history)
    idx = train_history[5].idxmax()  # max idx test acc
    model_best['score'].append(train_history[5][idx])
    # saves segments of data the model was trained with
    model_best['tra_with'].append(tra)
    model_best['val_with'].append(val)
    # save model initial weights
    model_best['initial_weights'].append(ini_wei)

    print(
        f'Best score fold {fn}: {hub.model.metrics_names[0]}: {train_history[4][idx]:.4}; {hub.model.metrics_names[1]}: {train_history[5][idx] * 100:.4}%')
    # Adds test score
    model_best['test_acc_per_fold'].append(train_history[5][idx] * 100)
    model_best['test_loss_per_fold'].append(train_history[4][idx])
    # confusion matrix per fold
    # -------------------------
    # Load best model in fold
    model_best_fold = tf.keras.models.load_model('log_savings/sleep_' + date + '/best_fold_model.h5')
    # Confusion matrix of best model in fold
    cm_per_fold.append(utils.get_confusion_matrix(model_best_fold, sleep.data['test'],
                                                  sleep.info['class_balance']['test']['value']))
    # Increase fold number
    fn += 1
# remove model per fold
os.remove('log_savings/sleep_' + date + '/best_fold_model.h5')
#%%
# ------------------------------------------------------------------------------------
#                                    Final Results
# ------------------------------------------------------------------------------------
# confusion matrix dataframe across participants
df = utils.cm_fold_to_df(cm_per_fold, model_best['test_loss_per_fold'])
utils.boxplot_evaluation_metrics_from_df(df, x_axes='fold')

# plots train history for the best model
utils.plot_train_test_history(model_best)

# Plots the confusion matrix of the best model the folds
cm_categories = {0: 'Conscious', 1: 'Unconscious'}
labels = [' True Pos', ' False Neg', ' False Pos', ' True Neg']
utils.make_confusion_matrix(cm_per_fold[np.argmax(model_best['score'])], group_names=labels, categories=cm_categories,
                            class_balance=sleep.info['class_balance']['test']['value'],
                            title='Confusion Matrix of Best Model')
#%%
# ------------------------------------------------------------------------------------
#                                       Savings
# ------------------------------------------------------------------------------------
# import json

df.to_csv('log_savings/sleep_' + date + '/folds.csv')
np.save('log_savings/sleep_' + date + '/model_best.npy', model_best)
# with open('log_savings/sleep_' + date + '/best_model.json', 'wb') as file:
#     file.write(json.dumps(model_best).encode("utf-8"))
    #json.dump(model_best, file, indent=4)


# utils.super_test(tf.keras.models.load_model('log_savings/sleep_' + date + '/all_folds_best_model.h5'),
#                  feature_function, dataset='anaesthesia')
# #%%
# # ------------------------------------------------------------------------------------
# #                                    Compares with Benchmark
# # ------------------------------------------------------------------------------------
# # check if current scores overpasses benchmark
# utils.check_benchmark(model_best, database='sleep')