# -----------------------------------------------------------------------------
#                           Libraries Needed
# -----------------------------------------------------------------------------

from tensorflow import keras

import seaborn as sns
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import modelhub as mh
import databases as dbs
from helpers_and_functions import main_functions as mpf

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
# loading trained model
trained_model = keras.models.load_model('log_savings/sleep_20210903_113105/all_folds_best_model.h5')
trained_model.summary()
# creating new model
new = mh.simple_cnn_2_1({'lr': 1e-6})
new.build_model_structure((25, 300, 2))
new.model.summary()
# Transferring weights from trained model to untrained model
mpf.transfer_model_weights(trained_model, new.model)
# compiling model
new.model.compile()
# model predictions
predictions = new.model.predict(sleep.data['test']['epochs'])

def predictions_to_df(predicted, labels, stats: bool = False):

    if not stats:
        df = pd.DataFrame()
        df = df.append([pd.DataFrame([np.append(labels[n], pred)]) for n, pred in enumerate(tqdm(predicted))], ignore_index=True)

    if stats:
        df = pd.DataFrame(columns=['mean', 'std', 'var', 'label'])
        for n, pred in enumerate(tqdm(predicted)):
            # calculates descriptive statistics
            row = np.array([np.mean(pred), np.std(pred), np.var(pred), labels[n]])
            # adds it to dataframe
            df = df.append(pd.DataFrame([row], columns=['mean', 'std', 'var', 'label']), ignore_index=True)
    return df

data_stats = predictions_to_df(predictions[0], sleep.data['test']['labels'], stats=True)
data_f = predictions_to_df(predictions[0], sleep.data['test']['labels'])
#pd.DataFrame([np.append(0, predictions[0][1])])

# #%%
#
# import matplotlib
# matplotlib.use('Qt5Agg')
# threedee = plt.figure().gca(projection='3d')
# threedee.scatter(data_stats['mean'], data_stats['std'], data_stats['var'], c=data_stats['label'], marker='o')
# threedee.set_xlabel('mean')
# threedee.set_ylabel('std')
# threedee.set_zlabel('var')
#
# plt.legend()
# plt.show()
#%%
sns.scatterplot(data=data_stats, x="std", y="mean", hue="label", style='label', size ="var")
plt.show()
#%%

sns.scatterplot(data=data_stats, x="var", y="mean", hue="label", style='label', size="std")
plt.show()
#%%

sns.scatterplot(data=data_stats, x="mean", y="std", hue="label", style='label', size="var")
plt.show()

# %%
plt.figure(figsize=[10, 5])
# plt.subplot(2, 1, 1)
plt.plot(predictions[1], '*', label='predictions')
# plt.subplot(2, 1, 1)
plt.plot(sleep.data['test']['labels'], '*', label='true label')
plt.legend()
plt.show()

# %%

# plot normal distribution
for n, pred in enumerate(tqdm(predictions[0])):
    sample = sorted(pred)
    pdf = stats.norm.pdf(sample, np.mean(sample), np.std(sample))
    if sleep.data['test']['labels'][n] == 0:
        plt.plot(sample, pdf, '--r')
    else:
        plt.plot(sample, pdf, '--b')
plt.show()
# %%
# plot std, mean
for n, pred in enumerate(tqdm(predictions[0])):

    if sleep.data['test']['labels'][n] == 0:
        plt.plot(np.std(sample), np.mean(sample), '*r')
    else:
        plt.plot(np.std(sample), np.mean(sample), '^b')

plt.xlabel('STD')
plt.ylabel('Mean')
plt.show()
# %%
from tqdm import tqdm

# plot normal distribution
for n, pred in enumerate(tqdm(predictions[0])):
    sample = sorted(pred)
    plt.scatter(np.std(sample), np.mean(sample), c=sleep.data['test']['labels'][n], cmap='brg')

plt.xlabel('STD')
plt.ylabel('Mean')
plt.colorbar()
plt.show()
