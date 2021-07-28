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
import utils
import config
import modelhub as mh
import databases as dbs
import functions_main as fm

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
sleep.transform(fm.nor_dataset)
# applying dataset transformation e.g. 'spectrogram'
sleep.transform(fm.raw_chunks_to_spectrograms, name='spectrogram')
# make dataset ready for training
sleep.get_ready_for_training()

# %%
# ------------------------------------------------------------------------------------
#                                Cross-Validation
# ------------------------------------------------------------------------------------
# Creates folder to store experiment
date = utils.create_date()
os.mkdir('savings/sleep_'+date)

# number of train epochs
train_epochs = 5

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                              verbose=1, restore_best_weights=True)

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
              'num_filters': 5,
              'num_filters': 5,
              'kernel_size': 3,
              'dense_units': 10,
              'out_size': 1}

for tra, val in kfold.split(sleep.data['train']['epochs'], sleep.data['train']['labels']):
    # Call the hub
    hub = mh.simple_cnn(parameters)
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
    pre_score = [0.0, 0.80]
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fn} ...')
    # -----------------------------------------------------------------------------
    #                               Train-Test Loop
    # -----------------------------------------------------------------------------
    for n_ep in range(train_epochs):
        # Train the model
        train_score = hub.model.fit(sleep.data['train']['epochs'][tra],
                                    sleep.data['train']['labels'][tra],
                                    validation_data=(sleep.data['train']['epochs'][val],
                                                     sleep.data['train']['labels'][val]),
                                    epochs=1, callbacks=[early_stop])
        # Evaluates on Test set
        test_scores = hub.model.evaluate(sleep.data['test']['epochs'], sleep.data['test']['labels'])
        # saving train history
        train_score = list(np.concatenate(list(train_score.history.values())))
        train_score.extend(test_scores)
        train_history.append(train_score)

        # Stores the best model in the fold
        if test_scores[1] > pre_score[1]:
            print(f'new best score in the fold: {test_scores[1]:.4}')
            # saves best model
            hub.model.save('savings/sleep_'+ date +'/best_model.h5')
            # updating previous score
            pre_score = copy.deepcopy(test_scores)
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

    model_best['test_acc_per_fold'].append(train_history[5][idx] * 100)
    model_best['test_loss_per_fold'].append(train_history[4][idx])

    # Increase fold number
    fn += 1

# %%
# ------------------------------------------------------------------------------------
#                                  Results
# ------------------------------------------------------------------------------------
# save model name
model_best['name'] = hub.model.name

# print summary of cross-validation scores
utils.print_cross_validation_scores(model_best['test_acc_per_fold'],
                                    model_best['test_loss_per_fold'])
# plots train history
utils.plot_train_test_history(model_best)

# loads best model
model_best_fold = tf.keras.models.load_model('savings/'+date+'/best_model.h5')
# get indices for positives and negatives
idx_0, idx_1 = sleep.data['test']['labels'] == 0, sleep.data['test']['labels'] == 1
# Evaluates on Test set
test_scores_0 = model_best_fold.evaluate(sleep.data['test']['epochs'][idx_0], sleep.data['test']['labels'][idx_0])
test_scores_1 = model_best_fold.evaluate(sleep.data['test']['epochs'][idx_1], sleep.data['test']['labels'][idx_1])
# Calculating Confusion Matrix
tp = test_scores_1[1] * sleep.info['class_balance']['test']['value'][1]
fp = (1 - test_scores_1[1]) * sleep.info['class_balance']['test']['value'][1]
tn = (1 - test_scores_0[1]) * sleep.info['class_balance']['test']['value'][0]
fn = test_scores_0[1] * sleep.info['class_balance']['test']['value'][0]
cm = np.array([[tp, fn], [fp, tn]], dtype=int)

cm_plot_labels = ['Unconscious', 'Conscious']
utils.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')