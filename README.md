# MSc_Project_Dissertation

Code developments for the MSc Project and Dissertation

In this project, the feature extraction will always take place at the end of the data pipeline, in a phase called Data Transformation. This obviously required to have applied some initial-data-processing and having cutting a long EEG waveform into epochs, among other things.

We can start this excersice by loading some EEG data from our sleep and the anaesthesia dataset's. These loaded EEG data will be passed throught the data pipeline shown in figure 1. This contains four main blocks:

Data Preparation: Where we can do initial steps such as load the EEG files, reject bad channels or select the ones to be processed. The operations in this block are mainly implemented using the mne library.
Data Preprocessing: Where the main operation is the filtering of the signal. At the end of this process, we will always end up with Epochs of a specific time length, ready to be transformed. These operations are done also with mne.
Data Transformation: The first transformation of the signal is thenormalization.
Ready for Model:

![image](https://user-images.githubusercontent.com/15948497/127302467-ff0d8e68-1cdb-448a-a52c-0c80201ca763.png)
