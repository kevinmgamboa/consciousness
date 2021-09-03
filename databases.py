"""
Sleep Data
----------
Load and prepare the dataset. In the future this will aso creates different dataset fold to run different experiments of generalization error.
Created on Tue Mar  2 12:10:27 2021
@author: Kevin Machado Gamboa
"""
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from mne.datasets.sleep_physionet.age import fetch_data
# Personal Libraries
from helpers_and_functions import config, main_functions as mpf


# -----------------------------------------------------------------------------
#                             Sleep Dataset
# -----------------------------------------------------------------------------
class sleep:
    """Manage the Sleep dataset

                Parameters
                ----------
                None

                Returns
                -------
                data : dict
                    dictionary with eeg recordings
        """

    def __init__(self):
        # Initializing dataset
        self.data = None
        # Initializing information variable
        self.info = dict.fromkeys(['n_samples', 'eeg_time'])

    def load_raw_data(self, t_files: int = 31,
                      d_filter: bool = False,
                      epochs: bool = False):

        if t_files > 31:
            raise AssertionError("Number of files can NOT exceed 31")

        # Defining number of eeg recordings
        subjects = range(t_files)  # after 32 problems matching sleep stages
        recordings = [1]
        # Getting eeg file names
        fnames = fetch_data(subjects=subjects, recording=recordings, on_missing='warn')
        # Load all 31 recordings
        self.data = [mpf.load_sleep_physionet_raw(f[0], f[1]) for f in tqdm(fnames)]

        if d_filter:
            low, high = 1, 49
            print(f'filtering dataset from {low}Hz to {high}Hz .. ')
            # applies filter
            self.data = [file.load_data().filter(low, high, fir_design='firwin') for file in tqdm(self.data)]

        # Generates general loading information
        self.info = {'n_samples': sum([file.n_times for file in self.data]),
                     'eeg_time': sum([file.times[-1] for file in self.data])}

        if epochs:
            # initializing storage variables
            epoch, label = [], []
            print('Extracting epochs')
            for _, raw in enumerate(tqdm(self.data)):
                # extracts epochs but stages ['n2', 'n3', 'n4', 'W] with binary=True
                raw = mpf.extract_epochs(raw, chunk_duration=config.EPOCH_LENGTH, dataset='sleep', binary=True)
                # stores epochs and labels in data container
                epoch.append(raw[0])
                label.append(raw[1])
            # concatenating data and save in data
            self.data = {'epochs': np.concatenate(epoch),
                         'labels': np.concatenate(label)}
            # updating information
            self.info['n_samples'] = len(self.data['epochs'])

    def load_epochs_labels(self, t_files: int = 10,
                           n_test: float = 0.2,
                           scheme: str = 'bp'):
        """

        @param t_files: number of files to load
        @param n_test: number of files for testing
        @param scheme: type of training scheme,
            bp: between patient
            ip: inter patient
        @return:
        """
        if t_files > 31 or t_files <= 4:
            raise AssertionError("Number of files have to be between 5-31")

        # number of files for train
        n_train = t_files - round(t_files * n_test)

        # Initializing container for dataset
        self.data = {'train': {"epochs": [], "labels": []},
                     'test': {"epochs": [], "labels": []}
                     }
        # Defining number of eeg recordings
        subjects = range(t_files)  # after 32 problems matching sleep stages
        recordings = [1]
        # Getting eeg file names
        fnames = fetch_data(subjects=subjects, recording=recordings, on_missing='warn')
        # Load recordings
        for n, file in enumerate(tqdm(fnames)):
            # load the raw data
            raw = mpf.load_sleep_physionet_raw(file[0], file[1], crop_wake_mins=30)
            # applies filter
            raw.load_data().filter(1, 49, fir_design='firwin')
            # extracts epochs but stages ['n2', 'n3', 'n4', 'W] with binary=True
            raw = mpf.extract_epochs(raw, chunk_duration=config.EPOCH_LENGTH, dataset='sleep', binary=True)

            if scheme == 'bp':
                # Applies between patient training method
                if n <= n_train:
                    # stores training epochs and labels in data container
                    self.data['train']["epochs"].append(raw[0])
                    self.data['train']["labels"].append(raw[1])
                else:
                    # stores test epochs and labels in data container
                    self.data['test']["epochs"].append(raw[0])
                    self.data['test']["labels"].append(raw[1])

            elif scheme == 'ip':
                # splits dataset keeping class distribution
                train_x, test_x, train_y, test_y = train_test_split(raw[0], raw[1], test_size=n_test)
                # stores training epochs and labels in data container
                self.data['train']["epochs"].append(train_x)
                self.data['train']["labels"].append(train_y)
                # stores test epochs and labels in data container
                self.data['test']["epochs"].append(test_x)
                self.data['test']["labels"].append(test_y)

        # concatenating epochs and labels in one ndarray
        self.data['train']["labels"] = np.concatenate(self.data['train']["labels"])
        self.data['train']["epochs"] = np.concatenate(self.data['train']["epochs"])
        self.data['test']['labels'] = np.concatenate(self.data['test']["labels"])
        self.data['test']['epochs'] = np.concatenate(self.data['test']["epochs"])
        # information
        self.info['n_samples'] = {'train': len(self.data['train']['labels']),
                                  'test': len(self.data['test']['labels'])}
        # time information
        self.info['eeg_time'] = {'total': mpf.eeg_time(sum(self.info['n_samples'].values()) * config.EPOCH_LENGTH),
                                 'train': mpf.eeg_time(list(self.info['n_samples'].values())[0] * config.EPOCH_LENGTH),
                                 'test': mpf.eeg_time(list(self.info['n_samples'].values())[1] * config.EPOCH_LENGTH)}
        # class balance information
        self.info['class_balance'] = {'train':
                                          {'value': dict(pd.Series(self.data['train']['labels']).value_counts()),
                                           'percentage': dict(pd.Series(self.data['train']['labels']).value_counts() /
                                                              self.info['n_samples']['train'])
                                           },
                                      'test':
                                          {'value': dict(pd.Series(self.data['test']['labels']).value_counts()),
                                           'percentage': dict(pd.Series(self.data['test']['labels']).value_counts() /
                                                              self.info['n_samples']['test'])}}

    def transform(self, my_function, name='unspecify'):
        print('transforming training dataset')
        # transforming training dataset
        self.data['train']['epochs'] = my_function(self.data['train']['epochs'])
        print('transforming test dataset')
        # transforming validation dataset
        self.data['test']['epochs'] = my_function(self.data['test']['epochs'])
        # information: Data Shape
        self.info['data_shape'] = np.shape(self.data['test']['epochs'][0])
        # name for transformation
        self.info['transformation'] = name

    def get_ready_for_training(self):
        self.data['train']['epochs'] = np.expand_dims(self.data['train']['epochs'], 1)
        self.data['test']['epochs'] = np.expand_dims(self.data['test']['epochs'], 1)

    def get_binary_labels(self):
        # replace sleep state N2 to unconscious=1
        self.data['train']["labels"] = pd.Series(self.data['train']['labels']).replace(2, 1).replace(3, 1).to_numpy()
        self.data['test']["labels"] = pd.Series(self.data['test']['labels']).replace(2, 1).replace(3, 1).to_numpy()
        # class balance information
        self.info['class_balance'] = {'train':
                                          {'value': dict(pd.Series(self.data['train']['labels']).value_counts()),
                                           'percentage': dict(pd.Series(self.data['train']['labels']).value_counts() /
                                                              self.info['n_samples']['train'])
                                           },
                                      'test':
                                          {'value': dict(pd.Series(self.data['test']['labels']).value_counts()),
                                           'percentage': dict(pd.Series(self.data['test']['labels']).value_counts() /
                                                              self.info['n_samples']['test'])}}

        print('sleep dataset with binary labels: [0=>conscious, 1=>unconscious]')
